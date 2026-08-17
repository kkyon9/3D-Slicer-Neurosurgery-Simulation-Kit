# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
"""脑牵拉模拟2.0（BrainRetractionSim2）—— 3D Slicer 脚本模块

功能：在 3D Slicer 中对脑表面模型进行实时牵拉形变模拟。
原理：
  - 使用 PBD 约束：压板刚性接触、边长约束、局部面积约束和弱全局体积约束。

特点：
  - 不需要安装任何额外依赖（只用 Slicer 自带的 numpy / VTK）
  - 不会修改原始模型，模拟在自动生成的"工作网格"上进行
  - 可在 3D 视图中直接拖动压板平面，实时改变牵拉位置和方向
  - 可按位移大小对脑表面染色，直观看到牵拉影响范围
  - 支持直接选择 Segmentation 节点作为输入（自动导出闭合表面）

安装与使用方法见同目录《使用说明.md》。
"""

import numpy as np
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin


# 脑实质默认色（灰粉/淡棕）：未染色时的模型颜色，也是位移色标 0 位移端的颜色
BRAIN_COLOR = (0.84, 0.71, 0.64)


# ============================================================================
# 模块注册信息
# ============================================================================

class BrainRetractionSim2(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "脑牵拉模拟2.0"
        self.parent.categories = ["神经外科模拟手术套件"]
        self.parent.dependencies = []
        self.parent.contributors = ["myz"]
        self.parent.helpText = (
            "在脑表面模型上实时模拟脑压板牵拉造成的软组织形变。\n\n"
            "使用步骤：\n"
            "① 选择脑表面模型或分割，点击“生成模拟网格”（自动识别输入类型）；\n"
            "② 新建压板平面（可在 3D 视图拖动/旋转平面调整位置与方向），设置压板宽度、长度与压入深度；\n"
            "③ 点击“开始形变”，用 PBD 物理约束模拟组织接触、体积保持与周围隆起。\n\n"
            "说明：本模块为教学与术前规划的定性模拟工具，结果不构成精确生物力学预测。"
        )
        self.parent.acknowledgementText = "基于 PBD 的实时软组织近似模拟（脑牵拉模拟2.0）。"


# ============================================================================
# 物理核心：纯 numpy 的软组织形变核心
# ============================================================================

class RetractionSimCore:
    """基于 PBD 的三角网格软组织形变核心。

    约束包括：刚性压板接触、边长约束、封闭表面体积约束。

    单位：毫米。所有数组均为 numpy。
    """

    GLOBAL_VOLUME_STRENGTH = 0.2

    def __init__(self):
        self.ready = False
        self._last_disp = None
        self.rest_normals = None
        self.vertex_areas = None
        self.volume_gradient = None
        self.edges = None
        self.edge_rest_len = None
        self.rest_tri_areas = None

    @staticmethod
    def _accumulate_vectors(indices, vectors, n_points):
        """按顶点索引累加三维修正量；bincount 比 np.add.at 更适合 PBD 热循环。"""
        return np.column_stack((
            np.bincount(indices, weights=vectors[:, 0], minlength=n_points),
            np.bincount(indices, weights=vectors[:, 1], minlength=n_points),
            np.bincount(indices, weights=vectors[:, 2], minlength=n_points),
        ))

    @staticmethod
    def _normalize_plate(plate):
        """把压板参数转换为 numpy 数组并归一化方向向量。"""
        center, direction, widthAxis, lengthAxis, width, length, thickness = plate
        c = np.asarray(center, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d /= (np.linalg.norm(d) + 1e-12)
        wa = np.asarray(widthAxis, dtype=np.float64)
        wa /= (np.linalg.norm(wa) + 1e-12)
        la = np.asarray(lengthAxis, dtype=np.float64)
        la /= (np.linalg.norm(la) + 1e-12)
        return c, d, wa, la, float(width), float(length), float(thickness)

    # ---------------- 初始化 ----------------

    def set_mesh(self, points, triangles):
        pts = np.asarray(points, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points 必须是 (N,3) 数组")
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError("triangles 必须是 (M,3) 数组")

        # 统一三角面朝向（保证有符号体积为正）
        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]
        vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v0).sum() / 6.0
        if vol < 0:
            tris = tris[:, [0, 2, 1]].copy()
            vol = -vol
        if abs(vol) < 1e-9:
            raise ValueError("网格体积接近 0，请确认模型是封闭的脑表面")

        self.rest_points = pts.copy()
        self.points = pts.copy()
        self.triangles = tris
        self.rest_volume = float(vol)

        self.fixed_mask = np.zeros(len(pts), dtype=bool)

        self.ready = True
        self._last_disp = None
        self.rest_normals = self._compute_vertex_normals(pts, tris)
        self.vertex_areas = self._compute_vertex_areas(pts, tris)
        self.volume_gradient = self._compute_volume_gradient(pts, tris)
        self.edges, self.edge_rest_len = self._compute_edges(pts, tris)
        e1 = pts[tris[:, 1]] - pts[tris[:, 0]]
        e2 = pts[tris[:, 2]] - pts[tris[:, 0]]
        self.rest_tri_areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    @staticmethod
    def _compute_vertex_normals(pts, tris):
        """由三角面计算单位顶点法线；网格朝向已在 set_mesh 中统一为向外。"""
        e1 = pts[tris[:, 1]] - pts[tris[:, 0]]
        e2 = pts[tris[:, 2]] - pts[tris[:, 0]]
        face_normals = np.cross(e1, e2)
        n = np.zeros_like(pts, dtype=np.float64)
        np.add.at(n, tris[:, 0], face_normals)
        np.add.at(n, tris[:, 1], face_normals)
        np.add.at(n, tris[:, 2], face_normals)
        norms = np.linalg.norm(n, axis=1)
        norms[norms < 1e-12] = 1.0
        n /= norms[:, None]
        return n

    @staticmethod
    def _compute_vertex_areas(pts, tris):
        """返回每个顶点的 Voronoi 近似面积（相邻三角形面积的 1/3）。"""
        e1 = pts[tris[:, 1]] - pts[tris[:, 0]]
        e2 = pts[tris[:, 2]] - pts[tris[:, 0]]
        tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        a = np.zeros(len(pts), dtype=np.float64)
        np.add.at(a, tris[:, 0], tri_area / 3.0)
        np.add.at(a, tris[:, 1], tri_area / 3.0)
        np.add.at(a, tris[:, 2], tri_area / 3.0)
        return a

    @staticmethod
    def _compute_volume_gradient(pts, tris):
        """返回封闭表面有符号体积对每个顶点位置的梯度。"""
        e1 = pts[tris[:, 1]] - pts[tris[:, 0]]
        e2 = pts[tris[:, 2]] - pts[tris[:, 0]]
        face_grad = np.cross(e1, e2) / 6.0
        grad = np.zeros_like(pts, dtype=np.float64)
        np.add.at(grad, tris[:, 0], face_grad)
        np.add.at(grad, tris[:, 1], face_grad)
        np.add.at(grad, tris[:, 2], face_grad)
        return grad

    @staticmethod
    def _compute_edges(pts, tris):
        """从三角面提取无重复边，并返回每条边的静止长度。"""
        edges = np.vstack([
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
        ])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0).astype(np.int64)
        rest_len = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)
        return edges, rest_len

    def _signed_volume(self, pts):
        """计算当前三角形网格的有符号体积。"""
        tris = self.triangles
        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]
        return np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v0).sum() / 6.0

    def reset(self):
        if not self.ready:
            return
        self.points[:] = self.rest_points
        self._last_disp = None

    # ---------------- 查询 ----------------

    def displacement(self):
        if self._last_disp is None:
            self._last_disp = np.linalg.norm(self.points - self.rest_points, axis=1)
        return self._last_disp

    # ---------------- 固定 ----------------

    def set_fixed_mask(self, mask):
        self.fixed_mask = np.asarray(mask, dtype=bool).copy()
        self.points[self.fixed_mask] = self.rest_points[self.fixed_mask]
        self._last_disp = None

    # ---------------- 快速直接形变 ----------------

    def solve(self, plate, iterations=30, edgeRelax=0.1, volumeStrength=1.0,
              influenceRadius=50.0):
        """用 PBD 求解当前压板接触下的软组织平衡形变。

        plate = (center, direction, widthAxis, lengthAxis, width, length, thickness)
          - center：压板贴合表面时的中心位置（未偏移）
          - direction：压板平面外法线方向
          - widthAxis / lengthAxis：压板面内两个轴
          - width / length：压板尺寸
          - thickness：压入深度
        """
        if not self.ready:
            return
        c, d, wa, la, width, length, adv = self._normalize_plate(plate)

        # 压板从贴合表面沿 -d 方向压入组织
        plate_center = c - d * adv
        half_w = float(width) / 2.0
        half_l = float(length) / 2.0

        p = self.rest_points.copy()
        user_fixed = self.fixed_mask
        rest_user_fixed = self.rest_points[user_fixed].copy()

        # 软影响半径：中心区域完全参与，边缘平滑过渡到不动
        dist_to_plate = np.linalg.norm(self.rest_points - plate_center, axis=1)
        transition = max(10.0, influenceRadius * 0.25)
        u = np.clip(
            (dist_to_plate - (influenceRadius - transition)) / (2.0 * transition),
            0.0,
            1.0,
        )
        influence_weight = 1.0 - u * u * (3.0 - 2.0 * u)

        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]
        rest_len = self.edge_rest_len
        t0 = self.triangles[:, 0]
        t1 = self.triangles[:, 1]
        t2 = self.triangles[:, 2]
        rest_area = self.rest_tri_areas
        n_points = len(p)

        vol_grad = self.volume_gradient.copy()
        vol_grad[user_fixed] = 0.0
        vol_denom = float((vol_grad * vol_grad).sum()) + 1e-12
        global_volume_strength = self.GLOBAL_VOLUME_STRENGTH

        for _ in range(max(1, int(iterations))):
            # 刚性压板接触
            along = p @ d - np.dot(plate_center, d)
            along_w = p @ wa - np.dot(plate_center, wa)
            along_l = p @ la - np.dot(plate_center, la)
            edge_margin = 2.0  # mm，压板边缘外扩，避免组织从压板边缘反折
            inside = (
                (np.abs(along_w) <= half_w + edge_margin)
                & (np.abs(along_l) <= half_l + edge_margin)
                & (along > 0.0)
            )
            if np.any(inside):
                p[inside] -= d[None, :] * along[inside, None]
            p[user_fixed] = rest_user_fixed

            # 边长约束
            diff = p[e1] - p[e0]
            length = np.linalg.norm(diff, axis=1)
            length[length < 1e-12] = 1e-12
            corr = (
                (0.5 * edgeRelax * (length - rest_len) / length)[:, None]
                * diff
            )
            p += self._accumulate_vectors(e0, corr, n_points)
            p -= self._accumulate_vectors(e1, corr, n_points)
            p[user_fixed] = rest_user_fixed

            # 限制单条边最大拉伸
            diff = p[e1] - p[e0]
            length = np.linalg.norm(diff, axis=1)
            length[length < 1e-12] = 1e-12
            max_strain = 1.8
            max_len = rest_len * max_strain
            over = length > max_len
            if np.any(over):
                factor = 0.5 * (1.0 - max_len[over] / length[over])
                stretch_corr = diff[over] * factor[:, None]
                p += self._accumulate_vectors(e0[over], stretch_corr, n_points)
                p -= self._accumulate_vectors(e1[over], stretch_corr, n_points)
            p[user_fixed] = rest_user_fixed

            # 局部面积约束：近似局部不可压缩，避免远处被全局拉扯
            if volumeStrength > 0.0:
                p0 = p[t0]
                p1 = p[t1]
                p2 = p[t2]
                ee1 = p1 - p0
                ee2 = p2 - p0
                cr = np.cross(ee1, ee2)
                norm = np.linalg.norm(cr, axis=1)
                norm[norm < 1e-12] = 1e-12
                unit = cr / norm[:, None]
                area = 0.5 * norm
                c_area = area - rest_area
                g0 = 0.5 * np.cross(unit, p2 - p1)
                g1 = 0.5 * np.cross(unit, p0 - p2)
                g2 = 0.5 * np.cross(unit, p1 - p0)
                denom = (
                    (g0 * g0).sum(axis=1)
                    + (g1 * g1).sum(axis=1)
                    + (g2 * g2).sum(axis=1)
                    + 1e-12
                )
                lam = volumeStrength * c_area / denom
                p -= self._accumulate_vectors(t0, lam[:, None] * g0, n_points)
                p -= self._accumulate_vectors(t1, lam[:, None] * g1, n_points)
                p -= self._accumulate_vectors(t2, lam[:, None] * g2, n_points)
                p[user_fixed] = rest_user_fixed

            # 弱全局体积约束，只用于防止长时间漂移
            c_vol = self._signed_volume(p) - self.rest_volume
            p -= global_volume_strength * c_vol / vol_denom * vol_grad
            p[user_fixed] = rest_user_fixed

            # 软影响过渡：离压板越远越接近原始形态
            p -= self.rest_points
            p *= influence_weight[:, None]
            p += self.rest_points
            p[user_fixed] = rest_user_fixed

        self.points[:] = p
        self._last_disp = None

    def fast_preview(self, plate, falloff=12.0, volumeCompensation=0.8):
        """单帧直接近似形变，用于拖动压板时快速预览。"""
        if not self.ready:
            return
        c, d, wa, la, width, length, adv = self._normalize_plate(plate)
        if width < 1e-6 or adv < 1e-6:
            self.reset()
            return

        def smoothstep(t):
            return t * t * (3.0 - 2.0 * t)

        x = self.rest_points
        along = x @ d - np.dot(c, d)
        along_w = x @ wa - np.dot(c, wa)
        along_l = x @ la - np.dot(c, la)
        half_w = float(width) / 2.0
        half_l = float(length) / 2.0
        edge_f = 2.0

        u_w = np.clip(np.abs(along_w) / max(half_w + edge_f, 1e-6), 0.0, 1.0)
        arc_w = np.sqrt(np.maximum(1.0 - u_w * u_w, 0.0))
        t_l = np.clip(
            (np.abs(along_l) - 0.75 * half_l) / max(0.25 * half_l + edge_f, 1e-6),
            0.0,
            1.0,
        )
        arc_l = 1.0 - smoothstep(t_l)
        plane_factor = arc_w * arc_l

        # 使用未偏移的压板中心，depth 直接以 -along 表示组织侧深度
        depth = -along
        t_exit = np.clip((depth - adv) / max(falloff, 1e-6), 0.0, 1.0)
        depth_factor = 1.0 - smoothstep(t_exit)
        depth_factor = np.where(depth >= 0.0, depth_factor, 0.0)

        factor = plane_factor * depth_factor
        factor[self.fixed_mask] = 0.0

        n = self.rest_normals
        A = self.vertex_areas
        s = adv * factor
        disp = (-n) * s[:, None]

        if volumeCompensation > 0.0 and n is not None and A is not None:
            ring_denom_w = max(half_w + edge_f, 1e-6)
            ring_denom_l = max(half_l + edge_f, 1e-6)
            ur = np.sqrt(
                (along_w / ring_denom_w) ** 2
                + (along_l / ring_denom_l) ** 2
            )
            ring = np.exp(-((ur - 1.0) ** 2) / (2.0 * 0.6 * 0.6))
            ring = np.where(depth_factor > 0.0, ring, 0.0)
            ring[self.fixed_mask] = 0.0
            vol_loss = float((s * A).sum())
            ring_int = float((ring * A).sum())
            if ring_int > 1e-12:
                alpha = volumeCompensation * vol_loss / ring_int
                disp += alpha * ring[:, None] * n

        self.points[:] = x + disp
        self._last_disp = None



# ============================================================================
# 逻辑层：Slicer / VTK 与物理核心之间的桥
# ============================================================================

class BrainRetractionSim2Logic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.core = RetractionSimCore()
        self.outputModel = None     # 模拟工作网格（不碰原模型）
        self.retractorModel = None  # 牵拉器的可视化
        self.retractorVisible = False  # 压板三维示意默认隐藏（仅保留压板平面）
        self.colorNode = None
        self._comCache = None
        self._pointsArray = None
        self._displayNode = None
        self._colorRangeHigh = None
        self._displacementArray = None

    # ---------------- 网格准备 ----------------

    def _preparePolyData(self, poly):
        """对 PolyData 做三角化、清理、取最大连通域，返回可用网格。"""
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(tri.GetOutputPort())
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputConnection(clean.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        return conn.GetOutput()

    def getCenterOfMass(self):
        """返回工作网格的质心缓存（世界坐标），用于自动确定压入方向。"""
        return self._comCache

    def _computeCenterOfMass(self, poly):
        comFilter = vtk.vtkCenterOfMass()
        comFilter.SetInputData(poly)
        comFilter.SetUseScalarsAsWeights(False)
        comFilter.Update()
        return np.asarray(comFilter.GetCenter(), dtype=np.float64)

    def prepareFromModel(self, inputModel, targetPoints=4000):
        """把输入表面模型精简成适合实时模拟的工作网格。返回 (输出模型, 顶点数, 三角面数)。"""
        if inputModel is None:
            raise ValueError("请先选择输入模型")
        src = inputModel.GetPolyData()
        if src is None or src.GetNumberOfPoints() == 0:
            raise ValueError("所选模型没有网格数据")

        poly = self._preparePolyData(src)

        # 若原模型带父级变换，把变换“烘焙”进坐标，使模拟在世界坐标系进行
        parent = inputModel.GetParentTransformNode()
        if parent is not None:
            general = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parent, None, general)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(general)
            tf.SetInputData(poly)
            tf.Update()
            poly = tf.GetOutput()

        return self._buildWorkModel(poly, inputModel.GetName(), targetPoints)

    def prepareFromSegmentation(self, inputSegmentation, targetPoints=4000):
        """从 Segmentation 节点导出第一个 segment 的闭合表面，再生成工作网格。"""
        if inputSegmentation is None:
            raise ValueError("请先选择分割节点")
        segmentation = inputSegmentation.GetSegmentation()
        if segmentation.GetNumberOfSegments() == 0:
            raise ValueError("所选分割没有分段")

        segmentID = segmentation.GetNthSegmentID(0)
        segment = segmentation.GetSegment(segmentID)
        if segment is None:
            raise ValueError("无法获取第一个分段")

        # 优先获取 Closed surface 表示；不存在则尝试创建
        poly = segment.GetRepresentation("Closed surface")
        if poly is None:
            if not segmentation.ContainsRepresentation("Closed surface"):
                segmentation.CreateRepresentation("Closed surface")
            poly = segment.GetRepresentation("Closed surface")
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("无法从分割生成闭合表面，请检查分段是否有体积")

        poly = self._preparePolyData(poly)

        # 若分割带父级变换，烘焙进坐标
        parent = inputSegmentation.GetParentTransformNode()
        if parent is not None:
            general = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parent, None, general)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(general)
            tf.SetInputData(poly)
            tf.Update()
            poly = tf.GetOutput()

        name = inputSegmentation.GetName() + "_segment"
        return self._buildWorkModel(poly, name, targetPoints)

    def _buildWorkModel(self, poly, baseName, targetPoints):
        """精简网格并创建输出 Model 节点。"""
        self._comCache = None
        # 精简到目标点数附近（最多 3 轮）
        for _ in range(8):
            n = poly.GetNumberOfPoints()
            if n <= targetPoints * 1.2:
                break
            dec = vtk.vtkQuadricDecimation()
            dec.SetInputData(poly)
            dec.SetTargetReduction(min(0.9, 1.0 - float(targetPoints) / n))
            dec.VolumePreservationOn()
            tri2 = vtk.vtkTriangleFilter()
            tri2.SetInputConnection(dec.GetOutputPort())
            tri2.Update()
            poly = tri2.GetOutput()
            slicer.app.processEvents()

        work = vtk.vtkPolyData()
        work.DeepCopy(poly)
        work.GetPointData().SetNormals(None)  # 让显示管线每帧重算法线

        from vtk.util.numpy_support import vtk_to_numpy
        pts = vtk_to_numpy(work.GetPoints().GetData()).astype(np.float64).copy()
        tris = vtk_to_numpy(work.GetPolys().GetConnectivityArray()).astype(np.int64)
        if tris.size % 3 != 0:
            raise ValueError("网格三角化失败，请换一个更干净的模型")
        tris = tris.reshape(-1, 3).copy()

        if self.outputModel is not None and slicer.mrmlScene.IsNodePresent(self.outputModel):
            slicer.mrmlScene.RemoveNode(self.outputModel)
        name = baseName + "_模拟网格"
        self.outputModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        self.outputModel.SetAndObservePolyData(work)
        self.outputModel.CreateDefaultDisplayNodes()
        dn = self.outputModel.GetDisplayNode()
        dn.SetColor(*BRAIN_COLOR)  # 脑实质色（未染色时的默认颜色）
        dn.SetOpacity(1.0)
        dn.SetBackfaceCulling(0)
        dn.SetVisibility(True)
        self._displayNode = dn
        self._pointsArray = slicer.util.arrayFromModelPoints(self.outputModel)
        self._displacementArray = None

        self.core = RetractionSimCore()
        self.core.set_mesh(pts, tris)
        self._comCache = self._computeCenterOfMass(work)
        return self.outputModel, len(pts), len(tris)

    # ---------------- 固定 ----------------

    def setAutoFixed(self, fraction, refPoint):
        """自动固定：离参考点（牵拉器尖端）最远的 fraction 比例顶点。"""
        c = self.core
        if not c.ready:
            return 0
        ref = np.asarray(refPoint, dtype=np.float64)
        d = np.linalg.norm(c.rest_points - ref, axis=1)
        k = min(len(d), max(1, int(round(len(d) * fraction))))
        idx = np.argpartition(d, len(d) - k)[-k:]
        mask = np.zeros(len(d), dtype=bool)
        mask[idx] = True
        c.set_fixed_mask(mask)
        return int(mask.sum())

    # ---------------- 模拟与显示 ----------------

    def applyDirectDeform(self, plate, iterations=30, edgeRelax=0.1,
                          volumeStrength=1.0, influenceRadius=50.0,
                          colorByDisplacement=True):
        """用 PBD 求解一帧并更新显示。"""
        if not self.core.ready:
            return
        center, direction, widthAxis, lengthAxis, width, length, thickness = plate
        self.core.solve(
            plate,
            iterations=iterations,
            edgeRelax=edgeRelax,
            volumeStrength=volumeStrength,
            influenceRadius=influenceRadius,
        )
        # 视觉压板画在实际接触面上，厚度固定 5mm
        d = np.asarray(direction, dtype=np.float64)
        d /= (np.linalg.norm(d) + 1e-12)
        contact_center = np.asarray(center, dtype=np.float64) - d * float(thickness)
        self.updateRetractor(contact_center, direction, widthAxis, lengthAxis, width, length, 5.0)
        self.updateDisplay(colorByDisplacement)

    def applyFastPreview(self, plate, colorByDisplacement=True):
        """用单帧近似形变更新显示，用于拖动压板时保持流畅。"""
        if not self.core.ready:
            return
        center, direction, widthAxis, lengthAxis, width, length, thickness = plate
        self.core.fast_preview(plate)
        d = np.asarray(direction, dtype=np.float64)
        d /= (np.linalg.norm(d) + 1e-12)
        contact_center = np.asarray(center, dtype=np.float64) - d * float(thickness)
        self.updateRetractor(contact_center, direction, widthAxis, lengthAxis, width, length, 5.0)
        self.updateDisplay(colorByDisplacement)

    def resetSimulation(self, colorByDisplacement=True):
        if not self.core.ready:
            return
        self.core.reset()
        self.updateDisplay(colorByDisplacement)

    def updateDisplay(self, colorByDisplacement=True):
        if self.outputModel is None or not slicer.mrmlScene.IsNodePresent(self.outputModel):
            return
        arr = self._pointsArray
        if arr is None:
            arr = slicer.util.arrayFromModelPoints(self.outputModel)
            self._pointsArray = arr
        arr[:] = self.core.points
        slicer.util.arrayFromModelPointsModified(self.outputModel)

        dn = self._displayNode
        if dn is None:
            dn = self.outputModel.GetDisplayNode()
        if colorByDisplacement:
            poly = self.outputModel.GetPolyData()
            disp = self.core.displacement().astype(np.float32)
            from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
            da = self._displacementArray
            if da is None or da.GetNumberOfTuples() != len(disp):
                da = numpy_to_vtk(np.empty(len(disp), dtype=np.float32), deep=True)
                da.SetName("位移mm")
                poly.GetPointData().RemoveArray("位移mm")
                poly.GetPointData().AddArray(da)
                self._displacementArray = da
            vtk_to_numpy(da)[:] = disp
            da.Modified()
            poly.GetPointData().SetActiveScalars("位移mm")

            if (self.colorNode is None
                    or not slicer.mrmlScene.IsNodePresent(self.colorNode)
                    or self.colorNode.GetNumberOfColors() < 256):
                # 若场景里已有保存时损坏（0 色）的同名色表，先移除再重建；
                # 用节点级 SetColor 写入，确保保存场景时颜色能正确序列化
                # （旧写法只改 LUT 不改节点颜色存储，导致存成 0 色 + Bad table range）
                if self.colorNode is not None and slicer.mrmlScene.IsNodePresent(self.colorNode):
                    slicer.mrmlScene.RemoveNode(self.colorNode)
                self.colorNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLColorTableNode", "牵拉位移色标")
                # 自定义色标：0 位移 = 脑实质本色，位移增大过渡到红色
                self.colorNode.SetTypeToUser()
                self.colorNode.SetNumberOfColors(256)
                br, bg, bb = BRAIN_COLOR
                for i in range(256):
                    t = (i / 255.0) ** 0.8
                    self.colorNode.SetColor(
                        i, br + (1.0 - br) * t, bg + (0.25 - bg) * t,
                        bb + (0.15 - bb) * t, 1.0)
                self.colorNode.Modified()
            dn.SetAndObserveColorNodeID(self.colorNode.GetID())
            dn.SetScalarVisibility(True)
            dn.SetActiveScalarName("位移mm")
            hi = float(max(1.0, np.percentile(disp, 99)))
            if (self._colorRangeHigh is None
                    or abs(hi - self._colorRangeHigh) > max(0.5, self._colorRangeHigh * 0.05)):
                dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
                dn.SetScalarRange(0.0, hi)
                self._colorRangeHigh = hi
        else:
            dn.SetScalarVisibility(False)

    def updateRetractor(self, center, direction, widthAxis, lengthAxis, width, length, thickness):
        """画出与压板平面完全一致的扁平长方体脑压板（银灰色半透明，默认隐藏）。

        当前版本只保留压板平面作为牵拉器，三维示意模型默认不创建/不显示。
        """
        if not self.retractorVisible:
            return
        center = np.asarray(center, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        widthAxis = np.asarray(widthAxis, dtype=np.float64)
        lengthAxis = np.asarray(lengthAxis, dtype=np.float64)
        direction /= (np.linalg.norm(direction) + 1e-12)
        widthAxis /= (np.linalg.norm(widthAxis) + 1e-12)
        lengthAxis /= (np.linalg.norm(lengthAxis) + 1e-12)

        cube = vtk.vtkCubeSource()
        cube.SetXLength(float(width))
        cube.SetYLength(float(thickness))
        cube.SetZLength(float(length))
        cube.Update()

        # 由平面局部坐标轴直接构造变换矩阵：局部 X→宽度轴，Y→方向，Z→长度轴
        mat = vtk.vtkMatrix4x4()
        for i in range(3):
            mat.SetElement(i, 0, widthAxis[i])
            mat.SetElement(i, 1, direction[i])
            mat.SetElement(i, 2, lengthAxis[i])
            mat.SetElement(i, 3, center[i])
        mat.SetElement(3, 3, 1.0)
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)

        tfpd = vtk.vtkTransformPolyDataFilter()
        tfpd.SetInputData(cube.GetOutput())
        tfpd.SetTransform(transform)
        tfpd.Update()

        if self.retractorModel is None or not slicer.mrmlScene.IsNodePresent(self.retractorModel):
            self.retractorModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "脑压板")
            self.retractorModel.CreateDefaultDisplayNodes()
            dn = self.retractorModel.GetDisplayNode()
            dn.SetColor(0.75, 0.78, 0.80)  # 银灰色
            dn.SetOpacity(0.55)
        self.retractorModel.SetAndObservePolyData(tfpd.GetOutput())
        self.retractorModel.SetDisplayVisibility(self.retractorVisible)

    def setRetractorVisibility(self, visible):
        """显示或隐藏脑压板模型（不影响模拟计算）。"""
        self.retractorVisible = bool(visible)
        if self.retractorModel is not None and slicer.mrmlScene.IsNodePresent(self.retractorModel):
            self.retractorModel.SetDisplayVisibility(self.retractorVisible)

    def cleanup(self):
        scene = slicer.mrmlScene
        if scene is not None:
            for node in (self.outputModel, self.retractorModel, self.colorNode):
                if node is not None and scene.IsNodePresent(node):
                    scene.RemoveNode(node)
        self.outputModel = None
        self.retractorModel = None
        self.colorNode = None
        self._comCache = None
        self._pointsArray = None
        self._displayNode = None
        self._displacementArray = None


# ============================================================================
# 界面层
# ============================================================================

class BrainRetractionSim2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        # 先初始化自身属性：基类是无 Qt 的普通 Python 类，
        # 且基类初始化在 standalone 模式下会自动调用 setup()，
        # 属性必须先就位以免被覆盖。
        self.logic = BrainRetractionSim2Logic()
        self.planeUpdateTimer = None  # 拖动压板时的节流更新定时器
        self.settleTimer = None  # 拖动结束后运行完整 PBD 的定时器
        self.deformationEnabled = False
        self._planeDirty = False
        self._fixedRef = None
        self._suppressInteractive = False
        self._planeObserverTag = None
        self._observedPlaneNode = None
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    # ---------------- UI 搭建 ----------------

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # 拖动压板平面时用定时器节流，避免每个鼠标事件都完整重算一次 PBD
        self.planeUpdateTimer = qt.QTimer(self.parent)
        self.planeUpdateTimer.setInterval(50)
        self.planeUpdateTimer.timeout.connect(self._on_plane_update_tick)

        # 拖动停止后，再运行一次完整 PBD 得到最终形变
        self.settleTimer = qt.QTimer(self.parent)
        self.settleTimer.setSingleShot(True)
        self.settleTimer.setInterval(180)
        self.settleTimer.timeout.connect(self._on_settle_timeout)

        # ---- ① 数据准备 ----
        prepBox = ctk.ctkCollapsibleButton()
        prepBox.text = "① 数据准备"
        self.layout.addWidget(prepBox)
        form = qt.QFormLayout(prepBox)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLModelNode", "vtkMRMLSegmentationNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = True
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("选择要模拟的脑表面模型或分割（请先在分割模块中准备好）")
        form.addRow("脑表面数据：", self.inputSelector)

        self.pointsSpin = qt.QSpinBox()
        self.pointsSpin.setRange(1000, 20000)
        self.pointsSpin.setSingleStep(500)
        self.pointsSpin.setValue(4000)
        self.pointsSpin.setToolTip("模拟网格的目标顶点数。越大越细腻但越卡，建议 3000–6000。")
        form.addRow("目标点数：", self.pointsSpin)

        self.prepareButton = qt.QPushButton("生成模拟网格")
        self.prepareButton.setToolTip("自动识别模型/分割并精简成模拟工作网格（不会改动原模型，原模型/分割将被隐藏）")
        form.addRow(self.prepareButton)

        self.prepStatus = qt.QLabel("尚未生成。")
        self.prepStatus.setWordWrap(True)
        form.addRow(self.prepStatus)

        # ---- ② 牵拉器（压板平面）----
        retBox = ctk.ctkCollapsibleButton()
        retBox.text = "② 牵拉器（压板平面）"
        self.layout.addWidget(retBox)
        form = qt.QFormLayout(retBox)

        planeButtonRow = qt.QHBoxLayout()
        self.createPlaneButton = qt.QPushButton("新建压板平面")
        self.createPlaneButton.setToolTip("在模型顶部自动创建一个可拖动的压板平面（平面法向即为压缩方向）")
        planeButtonRow.addWidget(self.createPlaneButton)
        self.togglePlaneButton = qt.QPushButton("隐藏压板平面")
        self.togglePlaneButton.setToolTip("隐藏或重新显示压板平面（不影响模拟计算）")
        self.togglePlaneButton.setCheckable(True)
        planeButtonRow.addWidget(self.togglePlaneButton)
        form.addRow(planeButtonRow)

        self.planeSelector = slicer.qMRMLNodeComboBox()
        self.planeSelector.nodeTypes = ["vtkMRMLMarkupsPlaneNode"]
        self.planeSelector.selectNodeUponCreation = True
        self.planeSelector.addEnabled = True
        self.planeSelector.removeEnabled = False
        self.planeSelector.noneEnabled = True
        self.planeSelector.showHidden = False
        self.planeSelector.showChildNodeTypes = False
        self.planeSelector.setMRMLScene(slicer.mrmlScene)
        self.planeSelector.setToolTip("选择压板平面；可在 3D 视图中直接拖动平面中心或旋转平面改变压缩方向")
        form.addRow("压板平面：", self.planeSelector)

        self.radiusSlider = ctk.ctkSliderWidget()
        self.radiusSlider.minimum = 2.0
        self.radiusSlider.maximum = 40.0
        self.radiusSlider.value = 12.0
        self.radiusSlider.decimals = 0
        self.radiusSlider.suffix = " mm"
        self.radiusSlider.setToolTip("脑压板的宽度（横向跨度），对应压板横向宽度")
        form.addRow("压板宽度：", self.radiusSlider)

        self.lengthSlider = ctk.ctkSliderWidget()
        self.lengthSlider.minimum = 10.0
        self.lengthSlider.maximum = 100.0
        self.lengthSlider.value = 50.0
        self.lengthSlider.decimals = 0
        self.lengthSlider.suffix = " mm"
        self.lengthSlider.setToolTip("脑压板的长度（纵向跨度），沿压板平面长轴方向")
        form.addRow("压板长度：", self.lengthSlider)

        self.advanceSlider = ctk.ctkSliderWidget()
        self.advanceSlider.minimum = 0.0
        self.advanceSlider.maximum = 120.0
        self.advanceSlider.value = 0.0
        self.advanceSlider.decimals = 1
        self.advanceSlider.suffix = " mm"
        self.advanceSlider.setToolTip("压板从脑表面沿压缩方向（平面法向/轴线方向）压入脑组织的深度；0 时贴表面，增大则压入")
        form.addRow("压入深度：", self.advanceSlider)

        self.iterSpin = qt.QSpinBox()
        self.iterSpin.setRange(5, 200)
        self.iterSpin.setSingleStep(5)
        self.iterSpin.setValue(10)
        self.iterSpin.setToolTip("每次形变的 PBD 迭代次数：越大越接近平衡态，但越慢。")
        form.addRow("PBD 迭代次数：", self.iterSpin)

        self.edgeSlider = ctk.ctkSliderWidget()
        self.edgeSlider.minimum = 0.01
        self.edgeSlider.maximum = 0.15
        self.edgeSlider.value = 0.1
        self.edgeSlider.decimals = 2
        self.edgeSlider.singleStep = 0.05
        self.edgeSlider.setToolTip("边长保持强度：越大组织越不容易被拉伸或压缩，边缘更稳定。")
        form.addRow("边长保持：", self.edgeSlider)

        self.volumeSlider = ctk.ctkSliderWidget()
        self.volumeSlider.minimum = 0.0
        self.volumeSlider.maximum = 1.0
        self.volumeSlider.value = 1.0
        self.volumeSlider.decimals = 2
        self.volumeSlider.singleStep = 0.05
        self.volumeSlider.setToolTip("体积保持强度：越大越接近不可压缩，压板周围隆起越明显。")
        form.addRow("体积保持：", self.volumeSlider)

        self.influenceSlider = ctk.ctkSliderWidget()
        self.influenceSlider.minimum = 10.0
        self.influenceSlider.maximum = 120.0
        self.influenceSlider.value = 50.0
        self.influenceSlider.decimals = 0
        self.influenceSlider.suffix = " mm"
        self.influenceSlider.setToolTip("只允许距离压板中心这么远的顶点参与形变，防止远处组织被拉扯。")
        form.addRow("影响半径：", self.influenceSlider)

        hint = qt.QLabel("提示：平面中心即压板中心，平面法向决定压入侧，平面大小即压板大小。PBD 会根据边长和体积约束把接触变形传播到周围组织。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)

        # ---- ③ 模拟 ----
        simBox = ctk.ctkCollapsibleButton()
        simBox.text = "③ 模拟"
        self.layout.addWidget(simBox)
        form = qt.QFormLayout(simBox)

        # 操作按钮排成一行，界面更紧凑
        simButtonRow = qt.QHBoxLayout()
        self.startButton = qt.QPushButton("开始形变")
        self.startButton.setCheckable(True)
        simButtonRow.addWidget(self.startButton)
        self.resetButton = qt.QPushButton("复位到初始形态")
        simButtonRow.addWidget(self.resetButton)
        form.addRow(simButtonRow)

        self.colorCheck = qt.QCheckBox("按位移染色（红=位移大，本色=不动）")
        self.colorCheck.setChecked(True)
        form.addRow(self.colorCheck)

        self.fastPreviewCheck = qt.QCheckBox("拖动压板时实时刷新形变（更耗性能）")
        self.fastPreviewCheck.setChecked(False)
        form.addRow(self.fastPreviewCheck)

        self.simStatus = qt.QLabel("—")
        self.simStatus.setWordWrap(True)
        form.addRow(self.simStatus)

        note = qt.QLabel("说明：本工具为定性模拟，用于理解牵拉对脑组织的影响趋势，"
                         "不构成精确生物力学预测，不能用于实际手术决策。")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)

        self.layout.addStretch(1)

        # ---- 信号连接 ----
        self.prepareButton.clicked.connect(self._on_prepare)
        self.createPlaneButton.clicked.connect(self._on_create_plane)
        self.togglePlaneButton.toggled.connect(self._on_toggle_plane)
        self.startButton.toggled.connect(self._on_start_toggled)
        self.resetButton.clicked.connect(self._on_reset)
        self.colorCheck.toggled.connect(self._on_color_toggled)

        # 参数变化实时响应
        self.advanceSlider.valueChanged.connect(self._on_interactive_changed)
        # 尺寸滑块变化时先同步压板平面尺寸，再重新计算形变
        self.radiusSlider.valueChanged.connect(self._on_size_slider_changed)
        self.lengthSlider.valueChanged.connect(self._on_size_slider_changed)
        self.iterSpin.valueChanged.connect(self._on_param_changed)
        self.edgeSlider.valueChanged.connect(self._on_interactive_changed)
        self.volumeSlider.valueChanged.connect(self._on_interactive_changed)
        self.influenceSlider.valueChanged.connect(self._on_interactive_changed)
        self.planeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._on_plane_node_changed)

    def cleanup(self):
        if self.planeUpdateTimer is not None:
            self.planeUpdateTimer.stop()
        if self.settleTimer is not None:
            self.settleTimer.stop()
        self._remove_plane_observer()
        self.removeObservers()
        self.logic.cleanup()

    # ---------------- 槽函数 ----------------

    def _on_param_changed(self):
        """任何形变参数变化时立即更新形变。"""
        if not self.logic.core.ready:
            return
        if not self.deformationEnabled:
            return
        plate = self._current_plate()
        if plate is None:
            return
        self._apply_fixation(plate)
        self._update_direct_deform(plate=plate)

    def _on_interactive_changed(self):
        """连续滑块变化时走节流通道，避免每次 valueChanged 都完整求解。"""
        if self._suppressInteractive:
            return
        if not self.logic.core.ready or not self.deformationEnabled:
            return
        self._planeDirty = True
        if self.fastPreviewCheck.isChecked():
            if not self.planeUpdateTimer.isActive():
                self.planeUpdateTimer.start()
        self.settleTimer.start()

    def _on_size_slider_changed(self):
        """尺寸滑块变化时同步更新当前压板平面的尺寸。"""
        planeNode = self.planeSelector.currentNode()
        if planeNode is None:
            return
        try:
            planeNode.SetSize(float(self.radiusSlider.value), float(self.lengthSlider.value))
        except Exception:
            pass
        self._on_interactive_changed()

    def _on_plane_node_changed(self):
        """压板平面节点切换时重新挂接事件监听。"""
        self._remove_plane_observer()
        planeNode = self.planeSelector.currentNode()
        if planeNode is not None:
            self._observedPlaneNode = planeNode
            self._planeObserverTag = planeNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._on_plane_modified)
        # 立即更新
        if self.logic.core.ready and self.deformationEnabled:
            plate = self._current_plate()
            if plate is None:
                return
            self._apply_fixation(plate)
            self._update_direct_deform(plate=plate)

    def _remove_plane_observer(self):
        if self._planeObserverTag is not None and self._observedPlaneNode is not None:
            self._observedPlaneNode.RemoveObserver(self._planeObserverTag)
        self._planeObserverTag = None
        self._observedPlaneNode = None

    def _on_plane_modified(self, caller, event):
        """压板平面被拖动时实时更新形变。"""
        if not self.logic.core.ready or not self.deformationEnabled:
            return
        self._planeDirty = True
        if self.fastPreviewCheck.isChecked():
            if not self.planeUpdateTimer.isActive():
                self.planeUpdateTimer.start()
        self.settleTimer.start()

    def _on_plane_update_tick(self):
        """节流后执行一次拖动更新，合并密集的鼠标事件。"""
        if not self._planeDirty:
            self.planeUpdateTimer.stop()
            return
        self._planeDirty = False
        if not self.logic.core.ready:
            self.planeUpdateTimer.stop()
            return
        if not self.deformationEnabled:
            self.planeUpdateTimer.stop()
            return
        if not self.fastPreviewCheck.isChecked():
            self.planeUpdateTimer.stop()
            return
        plate = self._current_plate()
        if plate is None:
            self.planeUpdateTimer.stop()
            return
        self._apply_fixation(plate)
        self.logic.applyFastPreview(plate, colorByDisplacement=self.colorCheck.isChecked())

    def _on_settle_timeout(self):
        """拖动停止后运行完整 PBD，得到最终物理形变。"""
        if not self.logic.core.ready or not self.deformationEnabled:
            return
        plate = self._current_plate()
        if plate is None:
            return
        self._apply_fixation(plate)
        self._update_direct_deform(plate=plate)

    def _on_prepare(self):
        self.prepareButton.setEnabled(False)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            slicer.util.showStatusMessage("正在生成模拟网格…")
            slicer.app.processEvents()
            node = self.inputSelector.currentNode()
            if node is None:
                raise ValueError("请先选择模型或分割节点")
            if node.IsA("vtkMRMLSegmentationNode"):
                out, nPts, nTri = self.logic.prepareFromSegmentation(
                    node, int(self.pointsSpin.value))
            else:
                out, nPts, nTri = self.logic.prepareFromModel(
                    node, int(self.pointsSpin.value))
            node.SetDisplayVisibility(False)
            self.deformationEnabled = False
            self._planeDirty = False
            self.planeUpdateTimer.stop()
            self.settleTimer.stop()
            self.startButton.setChecked(False)
            self.startButton.setText("开始形变")
        except Exception as e:
            qt.QMessageBox.warning(None, "生成失败", str(e))
            return
        finally:
            self.prepareButton.setEnabled(True)
            qt.QApplication.restoreOverrideCursor()
            slicer.util.showStatusMessage("")
        kind = "分割" if node.IsA("vtkMRMLSegmentationNode") else "模型"
        self.prepStatus.setText(
            f"已自动识别为{kind}数据，生成：顶点 {nPts} 个，三角面 {nTri} 个。\n"
            "原数据已隐藏（需要时可在“数据”模块点眼睛图标恢复显示）。")

    def _on_create_plane(self):
        if self.logic.outputModel is None:
            qt.QMessageBox.information(None, "提示", "请先在第①步生成模拟网格。")
            return
        b = self.logic.outputModel.GetPolyData().GetBounds()
        cx = (b[0] + b[1]) / 2.0
        cy = (b[2] + b[3]) / 2.0
        ztop = b[5]
        planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", "压板平面")
        planeNode.CreateDefaultDisplayNodes()
        # 平面中心放在模型顶部表面，法向朝下（指向脑组织内部）
        planeNode.SetCenter(cx, cy, ztop - 5.0)
        planeNode.SetNormal(0.0, 0.0, -1.0)
        width = float(self.radiusSlider.value)
        length = float(self.lengthSlider.value)
        planeNode.SetSize(width, length)
        try:
            dn = planeNode.GetDisplayNode()
            if dn is not None:
                dn.SetColor(1.0, 0.8, 0.1)
                dn.SetSelectedColor(1.0, 0.8, 0.1)
                dn.SetOpacity(0.6)
                dn.SetGlyphScale(1.5)
        except Exception:
            pass
        self.planeSelector.setCurrentNode(planeNode)
        planeNode.SetDisplayVisibility(True)
        self.togglePlaneButton.setChecked(False)
        self.togglePlaneButton.setText("隐藏压板平面")

    def _on_toggle_plane(self, checked):
        planeNode = self.planeSelector.currentNode()
        if planeNode is not None:
            planeNode.SetDisplayVisibility(not checked)
        self.togglePlaneButton.setText("显示压板平面" if checked else "隐藏压板平面")

    def _current_plate(self, advance=None):
        """由压板平面计算当前脑压板参数。

        返回 (center, direction, widthAxis, lengthAxis, width, length, thickness)：
          - center：压板贴合表面时的中心位置（未偏移）
          - direction：压板平面法向（用于确定受压侧与纵向深度分布）
          - widthAxis / lengthAxis：平面自身的两个面内坐标轴
          - width / length：压板横向宽度与纵向长度（= 平面 size）
          - thickness：作用厚度（= 压入深度，0 时贴合表面、无形变）
        """
        if advance is None:
            advance = float(self.advanceSlider.value)
        adv = float(advance)
        planeNode = self.planeSelector.currentNode()
        if planeNode is None:
            return None

        center = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 0.0]
        axisX = [0.0, 0.0, 0.0]
        axisY = [0.0, 0.0, 0.0]
        axisZ = [0.0, 0.0, 0.0]
        size = [0.0, 0.0]
        planeNode.GetCenterWorld(center)
        planeNode.GetNormalWorld(normal)
        planeNode.GetAxesWorld(axisX, axisY, axisZ)
        planeNode.GetSizeWorld(size)

        d = np.asarray(normal, dtype=np.float64)
        nrm = np.linalg.norm(d)
        if nrm < 1e-6:
            return None
        d /= nrm
        center = np.asarray(center, dtype=np.float64)

        # 默认压入方向：指向远离脑模型质心的一侧（即向外牵拉/抬起组织，
        # 相当于原“反转压入方向”勾选后的方向），不再提供反转选项。
        com = self.logic.getCenterOfMass()
        if com is not None:
            if np.dot(com - center, d) >= 0:
                d = -d

        widthAxis = np.asarray(axisX, dtype=np.float64)
        lengthAxis = np.asarray(axisY, dtype=np.float64)
        widthAxis /= (np.linalg.norm(widthAxis) + 1e-12)
        lengthAxis /= (np.linalg.norm(lengthAxis) + 1e-12)
        width = float(size[0])
        length = float(size[1])

        # 平面中心即压板贴合表面的位置；adv 由 PBD 核心作为压入深度处理
        thickness = adv
        return (center, d, widthAxis, lengthAxis, width, length, thickness)

    def _apply_fixation(self, plate=None):
        """自动固定：固定离压板中心最远的 15% 顶点（后台执行，无需手动设置）。"""
        if not self.logic.core.ready:
            return False
        try:
            if plate is None:
                plate = self._current_plate()
            if plate is None:
                return False
            ref = np.asarray(plate[0], dtype=np.float64)
            if self._fixedRef is not None and np.linalg.norm(ref - self._fixedRef) < 1.0:
                return True
            self.logic.setAutoFixed(0.15, ref)
            self._fixedRef = ref
        except Exception:
            return False
        return True

    def _on_start_toggled(self, checked):
        """开启/关闭形变：开启后模型按当前压板参数发生形变。"""
        if checked:
            if self.logic.outputModel is None or not self.logic.core.ready:
                qt.QMessageBox.information(None, "提示", "请先在第①步生成模拟网格。")
                self.startButton.setChecked(False)
                return
            plate = self._current_plate()
            if plate is None:
                qt.QMessageBox.information(None, "提示", "请先创建压板平面（第②步）。")
                self.startButton.setChecked(False)
                return
            if not self._apply_fixation(plate):
                self.startButton.setChecked(False)
                return
            self.deformationEnabled = True
            self._update_direct_deform(plate=plate)
            self.startButton.setText("关闭形变")
            self.simStatus.setText("形变已开启。")
        else:
            self.deformationEnabled = False
            self._planeDirty = False
            self.planeUpdateTimer.stop()
            self.settleTimer.stop()
            self.logic.resetSimulation(self.colorCheck.isChecked())
            self.startButton.setText("开始形变")
            self.simStatus.setText("形变已关闭，模型恢复初始形态。")

    def _on_reset(self):
        self._suppressInteractive = True
        self.advanceSlider.value = 0.0
        self._suppressInteractive = False
        self.logic.resetSimulation(self.colorCheck.isChecked())
        self.simStatus.setText("已复位到初始形态。")

    def _on_color_toggled(self, checked):
        if self.logic.outputModel is not None and self.logic.core.ready:
            self.logic.updateDisplay(checked)

    def _update_direct_deform(self, advance=None, plate=None):
        """立即计算并显示形变。"""
        if plate is None:
            plate = self._current_plate(advance)
        if plate is None:
            return
        self.logic.applyDirectDeform(
            plate,
            iterations=int(self.iterSpin.value),
            edgeRelax=float(self.edgeSlider.value),
            volumeStrength=float(self.volumeSlider.value),
            influenceRadius=float(self.influenceSlider.value),
            colorByDisplacement=self.colorCheck.isChecked(),
        )
        dispMax = float(self.logic.core.displacement().max())
        cur = float(self.advanceSlider.value) if advance is None else float(advance)
        self.simStatus.setText(f"压入 {cur:.1f} mm ｜ 最大位移 {dispMax:.1f} mm")
