# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
"""脑实质造瘘模拟·体积渲染（BrainFistulaVRSim）—— 3D Slicer 脚本模块

功能：在 3D Slicer 中模拟"脑实质造瘘 / 手术通道建立"（体积渲染版）：
  - 用两个可拖动点（入口点 + 靶点）快速移动/旋转一条造瘘通道；
  - 通道截面支持：管状（圆形）、椭圆管、口小底大锥形、椭圆锥形（扇形视野）；
  - 把通道内部 MRI 体素置为背景值，生成"掩膜体积"并打开体积渲染，
    3D 视图中通道呈现为脑实质内的透明隧道；
  - 不修改原始 MRI，全部结果生成在新节点上。

原理说明：Slicer 5.12 的体积渲染只支持盒形（ROI）裁剪，不支持任意形状裁剪，
因此用"掩膜体积"方案等效实现：通道内体素改为背景值后，体积渲染自然把通道
渲染成透明，即脑实质里的一条可透视隧道。

架构：Core / Logic / Widget 三层，只依赖 Slicer 自带的 numpy / VTK。
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


# 通道示意模型颜色（绿色半透明）
CHANNEL_COLOR = (0.25, 0.80, 0.55)


# ============================================================================
# 模块注册信息
# ============================================================================

class BrainFistulaVRSim(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "脑实质造瘘模拟（体积渲染）"
        self.parent.categories = ["神经外科模拟手术套件"]
        self.parent.dependencies = []
        self.parent.contributors = ["myz"]
        self.parent.helpText = (
            "模拟脑实质造瘘 / 手术通道建立（体积渲染版）：用入口点与靶点两个标记点\n"
            "控制一条管状 / 椭圆 / 口小底大锥形通道，把通道内部 MRI 体素置为背景值\n"
            "并打开体积渲染，3D 视图中通道呈现为脑实质内的透明隧道。\n\n"
            "使用步骤：\n"
            "① 选择 MRI 体积，点击“新建两点通道”，在 3D 视图中拖动 入口点 与 靶点；\n"
            "② 选择通道形状并调整入口/出口半径、椭圆度、旋转角、延伸量；\n"
            "③ 点击“生成掩膜体积并打开 VR”查看透明隧道。\n\n"
            "说明：本模块为教学与术前规划的定性模拟工具，结果不构成精确生物力学预测。"
        )
        self.parent.acknowledgementText = "基于参数化通道与 MRI 掩膜体积渲染的造瘘模拟。"


# ============================================================================
# 核心：纯 numpy / VTK 的通道几何（与表面雕刻版共用同一套公式）
# ============================================================================

class FistulaChannelCore:
    """通道几何与内部判定的数值核心。单位：毫米。
    只依赖 numpy / VTK，不接触任何 MRML 节点，便于独立测试。"""

    @staticmethod
    def np_to_polydata(pts, tris):
        """由 (N,3) 顶点与 (M,3) 三角面构建 vtkPolyData（VTK 9.6 cell array 写法）。"""
        from vtk.util.numpy_support import numpy_to_vtk
        poly = vtk.vtkPolyData()
        vpts = vtk.vtkPoints()
        vpts.SetData(numpy_to_vtk(np.asarray(pts, dtype=np.float64), deep=True))
        poly.SetPoints(vpts)
        cells = vtk.vtkCellArray()
        conn = np.asarray(tris, dtype=np.int64).reshape(-1)
        offsets = np.arange(0, len(conn) + 1, 3, dtype=np.int64)
        cells.SetData(numpy_to_vtk(offsets, deep=True),
                      numpy_to_vtk(conn, deep=True))
        poly.SetPolys(cells)
        return poly

    @staticmethod
    def _channel_frame(entry, target, rot_deg):
        """返回通道入口、单位轴、长度及旋转后的两个截面轴。"""
        E = np.asarray(entry, dtype=np.float64)
        T = np.asarray(target, dtype=np.float64)
        d = T - E
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            raise ValueError("入口点与靶点重合，通道深度为 0")
        d /= length
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(d, up))) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        u = np.cross(up, d)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(d, u)
        rot = np.deg2rad(rot_deg)
        cos_rot = np.cos(rot)
        sin_rot = np.sin(rot)
        u2 = u * cos_rot + v * sin_rot
        v2 = -u * sin_rot + v * cos_rot
        return E, d, length, u2, v2

    @staticmethod
    def build_channel(entry, target, ra0, rb0, ra1, rb1,
                      margin=8.0, n_theta=28, n_rings=20,
                      cap_exit=True, rot_deg=0.0):
        """生成封闭通道表面。
        entry：入口点（脑表面）；target：靶点（通道底端）。
        ra0/rb0：入口椭圆截面长/短半径；ra1/rb1：出口（靶点处）截面长/短半径。
        margin：入口端向脑外延伸的长度；cap_exit：True 时通道底端封闭于靶点。
        rot_deg：椭圆截面绕通道轴线的旋转角。
        返回 (pts, tris)，朝向一致向外（有符号体积 > 0）。"""
        E, d, L, u2, v2 = FistulaChannelCore._channel_frame(
            entry, target, rot_deg)

        z_start = -margin
        z_end = L if cap_exit else L + margin
        zs = np.linspace(z_start, z_end, n_rings)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

        def interp(r0, r1):
            return np.where(zs <= 0.0, r0, np.where(zs >= L, r1,
                            r0 + (r1 - r0) * (zs / L)))

        ra = interp(ra0, ra1)
        rb = interp(rb0, rb1)

        # 整个网格一次向量化生成，避免实时拖动时的 Python 双重循环。
        cos_theta = np.cos(thetas)
        sin_theta = np.sin(thetas)
        bases = E + zs[:, None] * d
        rings = (
            bases[:, None, :]
            + (ra[:, None] * cos_theta)[..., None] * u2
            + (rb[:, None] * sin_theta)[..., None] * v2
        )
        pts = rings.reshape(-1, 3)

        ring_idx = np.repeat(np.arange(n_rings - 1, dtype=np.int64), n_theta)
        theta_idx = np.tile(np.arange(n_theta, dtype=np.int64), n_rings - 1)
        theta_next = (theta_idx + 1) % n_theta
        a = ring_idx * n_theta + theta_idx
        b = ring_idx * n_theta + theta_next
        c = (ring_idx + 1) * n_theta + theta_next
        d2 = (ring_idx + 1) * n_theta + theta_idx
        side_tris = np.empty((2 * len(a), 3), dtype=np.int64)
        side_tris[0::2] = np.column_stack((a, b, c))
        side_tris[1::2] = np.column_stack((a, c, d2))

        c0_idx = len(pts)
        c1_idx = c0_idx + 1
        pts = np.concatenate((pts, [E + z_start * d, E + z_end * d]), axis=0)
        j = np.arange(n_theta, dtype=np.int64)
        jn = (j + 1) % n_theta
        last = (n_rings - 1) * n_theta
        cap_tris = np.concatenate((
            np.column_stack((np.full(n_theta, c0_idx), jn, j)),
            np.column_stack((np.full(n_theta, c1_idx), last + j, last + jn)),
        ), axis=0).astype(np.int64, copy=False)
        tris = np.concatenate((side_tris, cap_tris), axis=0)

        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2p = pts[tris[:, 2]]
        vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2p - v0), v0).sum() / 6.0
        if vol < 0:
            tris = tris[:, [0, 2, 1]].copy()
        return pts, tris

    @staticmethod
    def channel_volume(ra0, rb0, ra1, rb1, depth):
        """口小底大椭圆截面的通道体积近似（线性插值截面）。"""
        return float(np.pi * depth * (ra0 * rb0 + (ra0 * rb1 + ra1 * rb0) / 2.0
                                      + ra1 * rb1) / 3.0)

    @staticmethod
    def points_inside_channel(points, entry, target,
                              ra0, rb0, ra1, rb1,
                              margin=8.0, cap_exit=True, rot_deg=0.0):
        """向量化判定一批点（RAS，单位 mm）是否位于通道内部。
        与 build_channel 使用同一套截面插值：z<0 段半径为入口半径，
        z∈[0,L] 线性过渡到出口半径；cap_exit=True 时通道止于靶点。
        返回与 points 等长的 bool 数组。"""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        try:
            E, d, L, u2, v2 = FistulaChannelCore._channel_frame(
                entry, target, rot_deg)
        except ValueError:
            return np.zeros(len(P), dtype=bool)

        rel = P - E
        z = rel @ d
        # u2/v2 与 d 正交，无需先构造 rho = rel - z*d 的 (N,3) 临时数组。
        pu = rel @ u2
        pv = rel @ v2
        t = np.clip(z, 0.0, L) / L
        ra = ra0 + (ra1 - ra0) * t
        rb = rb0 + (rb1 - rb0) * t
        z_end = L if cap_exit else L + margin
        return (z >= -margin) & (z <= z_end) & \
               ((pu / ra) ** 2 + (pv / rb) ** 2 <= 1.0)


# ============================================================================
# 逻辑层：Slicer / VTK 节点与核心之间
# ============================================================================

class BrainFistulaVRSimLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.channelModel = None         # 通道三维示意模型
        self.channelLine = None          # 两点通道线（入口→靶点）
        self.maskVolumeNode = None       # 掩膜体积（创建后原地更新，不反复新建）
        self._lastMaskIdx = None         # 上一帧通道体素索引，用于增量恢复

    # ---------------- 两点通道线 ----------------

    def createChannelLineFromVolume(self, volumeNode, depth_mm=40.0):
        """按 MRI 体积的 RAS 包围盒中心创建 入口点→靶点 的两点通道线。"""
        if volumeNode is None:
            raise ValueError("请先选择 MRI 体积")
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        cx = (bounds[0] + bounds[1]) / 2.0
        cy = (bounds[2] + bounds[3]) / 2.0
        ztop = bounds[5]
        lineNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", "造瘘通道(入口→靶点)")
        lineNode.CreateDefaultDisplayNodes()
        lineNode.AddControlPointWorld(vtk.vtkVector3d(cx, cy, ztop))
        lineNode.AddControlPointWorld(
            vtk.vtkVector3d(cx, cy, max(bounds[4], ztop - depth_mm)))
        try:
            lineNode.SetNthControlPointLabel(0, "入口")
            lineNode.SetNthControlPointLabel(1, "靶点")
        except Exception:
            pass
        try:
            dn = lineNode.GetDisplayNode()
            dn.SetPointLabelsVisibility(True)
            dn.SetGlyphScale(2.5)
            dn.SetTextScale(3.5)
            dn.SetColor(0.95, 0.6, 0.15)
            dn.SetSelectedColor(0.95, 0.6, 0.15)
            dn.SetLineColor(0.95, 0.6, 0.15)
            dn.SetOpacity(0.9)
        except Exception:
            pass
        lineNode.SetDisplayVisibility(True)
        self.channelLine = lineNode
        return lineNode

    # ---------------- 通道参数与几何 ----------------

    @staticmethod
    def shapeRadii(shape, ra0, ra1, ratio):
        """由形状模式 + 入口/出口半径 + 椭圆比计算 (ra0, rb0, ra1, rb1)。"""
        ra0 = float(ra0)
        ra1 = float(ra1)
        ratio = max(float(ratio), 1.0)
        if shape == "管状（圆形截面）":
            return ra0, ra0, ra0, ra0
        if shape == "椭圆管":
            return ra0, ra0 / ratio, ra0, ra0 / ratio
        if shape == "锥形·口小底大":
            return ra0, ra0, max(ra1, ra0), max(ra1, ra0)
        if shape == "椭圆锥形·口小底大":
            r1 = max(ra1, ra0)
            return ra0, ra0 / ratio, r1, r1 / ratio
        return ra0, ra0, ra0, ra0

    def currentChannelGeometry(self, lineNode):
        """由两点线节点计算通道几何，返回 (entry, target, depth) 或 None。"""
        if lineNode is None or lineNode.GetNumberOfControlPoints() < 2:
            return None
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        lineNode.GetNthControlPointPositionWorld(0, p0)
        lineNode.GetNthControlPointPositionWorld(1, p1)
        entry = np.asarray(p0, dtype=np.float64)
        target = np.asarray(p1, dtype=np.float64)
        depth = float(np.linalg.norm(target - entry))
        if depth < 1e-6:
            return None
        return entry, target, depth

    def buildChannelPolyData(self, lineNode, shape, ra0, ra1, ratio,
                             margin, capExit, rotDeg):
        geo = self.currentChannelGeometry(lineNode)
        if geo is None:
            return None
        entry, target, depth = geo
        ra0f, rb0f, ra1f, rb1f = self.shapeRadii(shape, ra0, ra1, ratio)
        pts, tris = FistulaChannelCore.build_channel(
            entry, target, ra0f, rb0f, ra1f, rb1f,
            margin=margin, cap_exit=capExit, rot_deg=rotDeg)
        return FistulaChannelCore.np_to_polydata(pts, tris)

    def updateChannelModel(self, lineNode, shape, ra0, ra1, ratio,
                           margin, capExit, rotDeg):
        """更新绿色半透明的通道示意模型几何（实时跟随两点与参数）。
        注意：只更新几何，不改变显示/隐藏状态（由 setChannelModelVisible 控制）。"""
        poly = self.buildChannelPolyData(lineNode, shape, ra0, ra1, ratio,
                                         margin, capExit, rotDeg)
        if poly is None:
            return None
        if self.channelModel is None or not slicer.mrmlScene.IsNodePresent(self.channelModel):
            self.channelModel = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "造瘘通道(示意)")
            self.channelModel.CreateDefaultDisplayNodes()
            dn = self.channelModel.GetDisplayNode()
            dn.SetColor(*CHANNEL_COLOR)
            dn.SetOpacity(0.35)
            dn.SetBackfaceCulling(0)
            dn.SetVisibility(0)  # 默认隐藏，用户需要时才显示
        self.channelModel.SetAndObservePolyData(poly)
        return self.channelModel

    def setChannelModelVisible(self, visible):
        """显式设置绿色通道示意模型的显示/隐藏（默认隐藏）。"""
        if self.channelModel is not None and slicer.mrmlScene.IsNodePresent(self.channelModel):
            self.channelModel.GetDisplayNode().SetVisibility(1 if visible else 0)
            return True
        return False

    # ---------------- 体积渲染造瘘（掩膜） ----------------

    def _channelVoxelMask(self, volumeNode, entry, target,
                          ra0f, rb0f, ra1f, rb1f,
                          margin, capExit, rotDeg):
        """在体积的体素网格上计算通道内部的布尔掩膜。
        先用 RAS 包围盒裁剪到 IJK 子块，再逐体素中心做向量化判定。
        返回 (idx1d, inside, bounds, n_masked)。"""
        image = volumeNode.GetImageData()
        nx, ny, nz = image.GetDimensions()
        ijk2ras = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijk2ras)
        m = np.zeros((4, 4))
        for r in range(4):
            for c in range(4):
                m[r, c] = ijk2ras.GetElement(r, c)
        m_inv = np.linalg.inv(m)

        E = np.asarray(entry, dtype=np.float64)
        T = np.asarray(target, dtype=np.float64)
        axis = T - E
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            return None, None, None, 0
        axis /= length
        # margin 只沿通道轴延伸；旧实现把它也加到三个径向方向，产生大量无效候选体素。
        start = E - axis * margin
        end = T if capExit else T + axis * margin
        radial_pad = max(ra0f, rb0f, ra1f, rb1f) + 2.0
        lo = np.minimum(start, end) - radial_pad
        hi = np.maximum(start, end) + radial_pad
        corners = np.array([
            [x, y, z, 1.0]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ], dtype=np.float64)
        ijk_c = corners @ m_inv.T
        i0 = max(0, int(np.floor(ijk_c[:, 0].min())))
        i1 = min(nx - 1, int(np.ceil(ijk_c[:, 0].max())))
        j0 = max(0, int(np.floor(ijk_c[:, 1].min())))
        j1 = min(ny - 1, int(np.ceil(ijk_c[:, 1].max())))
        k0 = max(0, int(np.floor(ijk_c[:, 2].min())))
        k1 = min(nz - 1, int(np.ceil(ijk_c[:, 2].max())))
        if i1 < i0 or j1 < j0 or k1 < k0:
            return None, None, None, 0

        # 用稀疏广播网格直接应用仿射矩阵，省去 (N,4) 齐次坐标大数组。
        kk, jj, ii = np.ogrid[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
        grid_shape = (k1 - k0 + 1, j1 - j0 + 1, i1 - i0 + 1)
        ras = np.empty(grid_shape + (3,), dtype=np.float64)
        for component in range(3):
            ras[..., component] = (
                m[component, 0] * ii
                + m[component, 1] * jj
                + m[component, 2] * kk
                + m[component, 3]
            )
        inside = FistulaChannelCore.points_inside_channel(
            ras.reshape(-1, 3), E, T, ra0f, rb0f, ra1f, rb1f,
            margin=margin, cap_exit=capExit, rot_deg=rotDeg)
        idx1d = ((kk * ny + jj) * nx + ii).ravel()
        return idx1d, inside, (i0, i1, j0, j1, k0, k1), int(inside.sum())

    def createFistulaMaskVolume(self, volumeNode, lineNode, shape, ra0, ra1,
                                ratio, margin, capExit, rotDeg,
                                backgroundValue=0.0, name="造瘘VR掩膜"):
        """深拷贝 MRI 体积，把通道内部体素置为背景值，生成用于体积渲染的掩膜体积。
        不修改原始体积。返回 (maskNode, nMasked)。"""
        if volumeNode is None:
            raise ValueError("请先选择 MRI 体积")
        if lineNode is None or lineNode.GetNumberOfControlPoints() < 2:
            raise ValueError("请先创建并调整两点通道")
        image = volumeNode.GetImageData()
        if image is None or image.GetPointData().GetScalars() is None:
            raise ValueError("所选体积没有图像数据")
        geo = self.currentChannelGeometry(lineNode)
        if geo is None:
            raise ValueError("两点重合，无法确定通道方向")
        entry, target, depth = geo
        ra0f, rb0f, ra1f, rb1f = self.shapeRadii(shape, ra0, ra1, ratio)

        idx1d, inside, bounds, nMasked = self._channelVoxelMask(
            volumeNode, entry, target, ra0f, rb0f, ra1f, rb1f,
            margin, capExit, rotDeg)
        if nMasked == 0:
            raise ValueError(
                "通道没有覆盖任何体素：请检查通道是否位于所选 MRI 体积范围内")

        # 校验通过后再替换旧掩膜（生成失败时保留上一次的掩膜和 VR）
        self.restoreVolumeRendering()

        from vtk.util.numpy_support import vtk_to_numpy
        newImg = vtk.vtkImageData()
        newImg.DeepCopy(image)
        arr = vtk_to_numpy(newImg.GetPointData().GetScalars())
        if arr.ndim == 2:
            arr = arr[:, 0]  # 只处理单通道标量（MRI 常规情况）
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            bg = int(np.clip(backgroundValue, info.min, info.max))
        else:
            bg = float(backgroundValue)
        arr[idx1d[inside]] = bg
        newImg.GetPointData().GetScalars().Modified()
        newImg.Modified()

        maskNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", name)
        maskNode.CopyOrientation(volumeNode)
        maskNode.SetAndObserveImageData(newImg)
        maskNode.CreateDefaultDisplayNodes()
        md = maskNode.GetDisplayNode()
        sd = volumeNode.GetDisplayNode()
        if md is not None and sd is not None:
            try:
                md.SetAutoWindowLevel(0)
                md.SetWindow(sd.GetWindow())
                md.SetLevel(sd.GetLevel())
            except Exception:
                pass
        maskNode.SetAttribute("BrainFistulaVRSim.VRMask", "1")
        maskNode.SetAttribute("BrainFistulaVRSim.VROriginalID", volumeNode.GetID())
        self.maskVolumeNode = maskNode
        self._lastMaskIdx = idx1d[inside].copy()
        return maskNode, int(nMasked)

    def updateFistulaMaskVolume(self, volumeNode, lineNode, shape, ra0, ra1,
                                ratio, margin, capExit, rotDeg,
                                backgroundValue=0.0):
        """在已创建的掩膜体积上原地刷新通道（拖动两点/调参后实时更新）。
        复用同一 vtkImageData：先把上一帧通道体素恢复为原始值，再把新通道置为背景值。
        返回 (maskNode, nMasked)；若掩膜尚不存在或换了体积则走创建流程。"""
        maskNode = self.maskVolumeNode
        if maskNode is None or not slicer.mrmlScene.IsNodePresent(maskNode):
            return self.createFistulaMaskVolume(
                volumeNode, lineNode, shape, ra0, ra1, ratio,
                margin, capExit, rotDeg, backgroundValue=backgroundValue)
        origID = maskNode.GetAttribute("BrainFistulaVRSim.VROriginalID")
        if volumeNode is None or origID != volumeNode.GetID():
            return self.createFistulaMaskVolume(
                volumeNode, lineNode, shape, ra0, ra1, ratio,
                margin, capExit, rotDeg, backgroundValue=backgroundValue)
        if lineNode is None or lineNode.GetNumberOfControlPoints() < 2:
            raise ValueError("请先创建并调整两点通道")
        geo = self.currentChannelGeometry(lineNode)
        if geo is None:
            raise ValueError("两点重合，无法确定通道方向")
        entry, target, depth = geo
        ra0f, rb0f, ra1f, rb1f = self.shapeRadii(shape, ra0, ra1, ratio)

        idx1d, inside, bounds, nMasked = self._channelVoxelMask(
            volumeNode, entry, target, ra0f, rb0f, ra1f, rb1f,
            margin, capExit, rotDeg)
        if nMasked == 0:
            raise ValueError(
                "通道没有覆盖任何体素：请检查通道是否位于所选 MRI 体积范围内")

        from vtk.util.numpy_support import vtk_to_numpy
        image = maskNode.GetImageData()
        arr = vtk_to_numpy(image.GetPointData().GetScalars())
        if arr.ndim == 2:
            arr = arr[:, 0]
        origArr = vtk_to_numpy(
            volumeNode.GetImageData().GetPointData().GetScalars())
        if origArr.ndim == 2:
            origArr = origArr[:, 0]

        # 恢复上一帧通道体素（掩膜图与原始图只差通道区域，可精确增量回滚）
        if self._lastMaskIdx is not None:
            arr[self._lastMaskIdx] = origArr[self._lastMaskIdx]

        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            bg = int(np.clip(backgroundValue, info.min, info.max))
        else:
            bg = float(backgroundValue)
        newIdx = idx1d[inside]
        arr[newIdx] = bg
        self._lastMaskIdx = newIdx.copy()
        image.GetPointData().GetScalars().Modified()
        image.Modified()
        maskNode.Modified()
        return maskNode, int(nMasked)

    def _findVRMaskNode(self):
        """按自定义属性找回本模块生成的掩膜体积（切换模块/重开面板后仍可恢复）。"""
        nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        for i in range(nodes.GetNumberOfItems()):
            n = nodes.GetItemAsObject(i)
            if n.GetAttribute("BrainFistulaVRSim.VRMask") == "1":
                return n
        return None

    def setupVolumeRendering(self, maskNode, originalVolume=None):
        """在掩膜体积上创建默认体积渲染显示并打开；隐藏原体积的 3D 显示。
        需要完整版 Slicer（含 Volume Rendering 模块），需在带 GUI 的会话中运行。"""
        if maskNode is None or not slicer.mrmlScene.IsNodePresent(maskNode):
            raise ValueError("掩膜体积不存在")
        try:
            volRenLogic = slicer.modules.volumerendering.logic()
        except Exception:
            raise ValueError(
                "体积渲染模块不可用：请确认 3D Slicer 完整安装（含 Volume Rendering）")
        origVR = None
        if originalVolume is not None:
            od = originalVolume.GetDisplayNode()
            if od is not None:
                od.SetVisibility3D(False)
            try:
                origVR = volRenLogic.GetFirstVolumeRenderingDisplayNode(originalVolume)
                if origVR is not None:
                    origVR.SetVisibility(False)
            except Exception:
                pass
        vr = volRenLogic.CreateDefaultVolumeRenderingNodes(maskNode)
        if vr is None:
            raise ValueError("创建体积渲染显示节点失败")
        # 继承原体积的渲染预设；没有则用 MRI 默认预设（低强度透明，保证隧道显示）
        try:
            preset = None
            if origVR is not None:
                preset = origVR.GetVolumePropertyNode()
            if preset is None:
                preset = volRenLogic.GetPresetByName("MR-Default")
            if preset is not None and vr.GetVolumePropertyNode() is not None:
                vr.GetVolumePropertyNode().Copy(preset)
        except Exception:
            pass
        vr.SetVisibility(True)
        return vr

    def restoreVolumeRendering(self, maskNode=None, originalVolume=None):
        """关闭掩膜体积渲染、移除掩膜体积、恢复原始体积的 3D 显示。"""
        if maskNode is None or not slicer.mrmlScene.IsNodePresent(maskNode):
            maskNode = self._findVRMaskNode()
        if maskNode is not None and slicer.mrmlScene.IsNodePresent(maskNode):
            if originalVolume is None:
                origID = maskNode.GetAttribute("BrainFistulaVRSim.VROriginalID")
                if origID:
                    originalVolume = slicer.mrmlScene.GetNodeByID(origID)
            try:
                volRenLogic = slicer.modules.volumerendering.logic()
                vr = volRenLogic.GetFirstVolumeRenderingDisplayNode(maskNode)
                if vr is not None:
                    vr.SetVisibility(False)
                    slicer.mrmlScene.RemoveNode(vr)
            except Exception:
                pass
            # 移除体积的普通显示节点（体积删除时不会自动带走显示节点）
            md = maskNode.GetDisplayNode()
            if md is not None and slicer.mrmlScene.IsNodePresent(md):
                slicer.mrmlScene.RemoveNode(md)
            slicer.mrmlScene.RemoveNode(maskNode)
        if originalVolume is not None:
            od = originalVolume.GetDisplayNode()
            if od is not None:
                od.SetVisibility3D(True)
        # 掩膜已移除，清空实时更新状态
        self.maskVolumeNode = None
        self._lastMaskIdx = None

    def cleanup(self):
        if self.channelModel is not None and slicer.mrmlScene.IsNodePresent(self.channelModel):
            slicer.mrmlScene.RemoveNode(self.channelModel)
        self.channelModel = None
        self.channelLine = None
        self.maskVolumeNode = None
        self._lastMaskIdx = None


# ============================================================================
# 界面层
# ============================================================================

class BrainFistulaVRSimWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        # 先初始化自身属性：基类是无 Qt 的普通 Python 类，
        # 且基类初始化在 standalone 模式下会自动调用 setup()，
        # 属性必须先就位以免被覆盖。（与脑牵拉模块同一模式）
        self.logic = BrainFistulaVRSimLogic()
        self._observedLineNode = None
        self._lineObserverTag = None
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    # ---------------- UI 搭建 ----------------

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # ---- ① MRI 体积与两点通道 ----
        dataBox = ctk.ctkCollapsibleButton()
        dataBox.text = "① MRI 体积与两点通道"
        self.layout.addWidget(dataBox)
        form = qt.QFormLayout(dataBox)

        self.mriSelector = slicer.qMRMLNodeComboBox()
        self.mriSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.mriSelector.selectNodeUponCreation = True
        self.mriSelector.addEnabled = False
        self.mriSelector.removeEnabled = False
        self.mriSelector.noneEnabled = True
        self.mriSelector.showHidden = False
        self.mriSelector.showChildNodeTypes = False
        self.mriSelector.setMRMLScene(slicer.mrmlScene)
        self.mriSelector.setToolTip("选择要观察的 MRI 体积（如 T1）。会生成一份“通道内体素置为背景值”的副本，原始体积不被修改")
        form.addRow("MRI 体积：", self.mriSelector)

        btnRow = qt.QHBoxLayout()
        self.createLineButton = qt.QPushButton("新建两点通道")
        self.createLineButton.setToolTip("按 MRI 体积中心创建 入口点 与 靶点 两个标记点")
        btnRow.addWidget(self.createLineButton)
        self.toggleLineButton = qt.QPushButton("隐藏通道线")
        self.toggleLineButton.setCheckable(True)
        self.toggleLineButton.setToolTip("隐藏或重新显示两点通道线")
        btnRow.addWidget(self.toggleLineButton)
        form.addRow(btnRow)

        self.lineSelector = slicer.qMRMLNodeComboBox()
        self.lineSelector.nodeTypes = ["vtkMRMLMarkupsLineNode"]
        self.lineSelector.selectNodeUponCreation = True
        self.lineSelector.addEnabled = True
        self.lineSelector.removeEnabled = False
        self.lineSelector.noneEnabled = True
        self.lineSelector.showHidden = False
        self.lineSelector.showChildNodeTypes = False
        self.lineSelector.setMRMLScene(slicer.mrmlScene)
        self.lineSelector.setToolTip("选择两点通道线；可在 3D 视图中直接拖动 入口点 / 靶点")
        form.addRow("通道线：", self.lineSelector)

        hint = qt.QLabel(
            "提示：在 3D 视图中直接拖动 入口点 / 靶点 即可移动/旋转通道；\n"
            "两点距离 = 通道深度，两点连线方向 = 通道方向。参数变化实时生效。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)

        # ---- ② 通道形状 ----
        shapeBox = ctk.ctkCollapsibleButton()
        shapeBox.text = "② 通道形状"
        self.layout.addWidget(shapeBox)
        form = qt.QFormLayout(shapeBox)

        self.shapeCombo = qt.QComboBox()
        self.shapeCombo.addItems(["管状（圆形截面）", "椭圆管",
                                  "锥形·口小底大", "椭圆锥形·口小底大"])
        self.shapeCombo.setToolTip(
            "口小底大：入口小、深部（靶点处）宽，模拟内镜/造瘘的扇形视野")
        form.addRow("通道形状：", self.shapeCombo)

        self.ra0Slider = ctk.ctkSliderWidget()
        self.ra0Slider.minimum = 0.5
        self.ra0Slider.maximum = 30.0
        self.ra0Slider.value = 4.0
        self.ra0Slider.decimals = 1
        self.ra0Slider.singleStep = 0.5
        self.ra0Slider.suffix = " mm"
        self.ra0Slider.setToolTip("入口处（脑表面）通道半径")
        form.addRow("入口半径：", self.ra0Slider)

        self.ra1Slider = ctk.ctkSliderWidget()
        self.ra1Slider.minimum = 0.5
        self.ra1Slider.maximum = 40.0
        self.ra1Slider.value = 12.0
        self.ra1Slider.decimals = 1
        self.ra1Slider.singleStep = 0.5
        self.ra1Slider.suffix = " mm"
        self.ra1Slider.setToolTip("出口处（靶点）通道半径；口小底大时大于入口半径")
        form.addRow("出口半径：", self.ra1Slider)

        self.ratioSlider = ctk.ctkSliderWidget()
        self.ratioSlider.minimum = 1.0
        self.ratioSlider.maximum = 5.0
        self.ratioSlider.value = 1.0
        self.ratioSlider.decimals = 1
        self.ratioSlider.singleStep = 0.1
        self.ratioSlider.setToolTip("截面椭圆度（长轴/短轴）；=1 为圆形，仅椭圆管/椭圆锥形生效")
        form.addRow("椭圆度：", self.ratioSlider)

        self.rotSlider = ctk.ctkSliderWidget()
        self.rotSlider.minimum = 0.0
        self.rotSlider.maximum = 180.0
        self.rotSlider.value = 0.0
        self.rotSlider.decimals = 0
        self.rotSlider.singleStep = 5.0
        self.rotSlider.suffix = "°"
        self.rotSlider.setToolTip("椭圆截面绕通道轴线旋转的角度（椭圆形状时生效）")
        form.addRow("截面旋转角：", self.rotSlider)

        self.marginSlider = ctk.ctkSliderWidget()
        self.marginSlider.minimum = 2.0
        self.marginSlider.maximum = 20.0
        self.marginSlider.value = 8.0
        self.marginSlider.decimals = 0
        self.marginSlider.suffix = " mm"
        self.marginSlider.setToolTip("通道向脑表面外延伸的长度（保证入口处完全挖空）")
        form.addRow("入口延伸量：", self.marginSlider)

        self.capExitCheck = qt.QCheckBox("底端封闭（盲端，通道止于靶点）")
        self.capExitCheck.setChecked(True)
        self.capExitCheck.setToolTip("勾选：通道到靶点即结束（盲端）；取消：贯通到靶点之外")
        form.addRow(self.capExitCheck)

        self.depthLabel = qt.QLabel("通道深度：-- mm")
        self.depthLabel.setWordWrap(True)
        form.addRow(self.depthLabel)

        self.toggleChannelModelButton = qt.QPushButton("显示通道模型")
        self.toggleChannelModelButton.setCheckable(True)
        self.toggleChannelModelButton.setToolTip(
            "绿色半透明通道模型默认隐藏；勾选后显示，便于对照通道位置")
        form.addRow(self.toggleChannelModelButton)

        # ---- ③ 体积渲染（掩膜） ----
        vrBox = ctk.ctkCollapsibleButton()
        vrBox.text = "③ 体积渲染（掩膜）"
        self.layout.addWidget(vrBox)
        form = qt.QFormLayout(vrBox)

        self.backgroundSpin = qt.QDoubleSpinBox()
        self.backgroundSpin.setRange(-2048.0, 4096.0)
        self.backgroundSpin.setDecimals(1)
        self.backgroundSpin.setValue(0.0)
        self.backgroundSpin.setToolTip(
            "通道内部体素被改成的数值。MRI 通常用 0（多数渲染预设中为透明）；"
            "若通道不透明请调小，如 -1000")
        form.addRow("通道背景值：", self.backgroundSpin)

        self.realtimeCheck = qt.QCheckBox("实时更新（拖动两点/调整参数自动刷新掩膜）")
        self.realtimeCheck.setChecked(True)
        self.realtimeCheck.setToolTip(
            "勾选后，拖动 入口点/靶点 或调整通道参数时，掩膜体积自动刷新，无需再点按钮；"
            "取消勾选则恢复为手动点按钮刷新")
        form.addRow(self.realtimeCheck)

        vrRow = qt.QHBoxLayout()
        self.vrCreateButton = qt.QPushButton("生成掩膜体积并打开 VR")
        self.vrCreateButton.setToolTip(
            "深拷贝 MRI，把通道内部体素置为背景值并打开体积渲染，"
            "在 3D 视图中呈现为脑实质内的透明隧道；首次生成后，实时更新由复选框控制")
        vrRow.addWidget(self.vrCreateButton)
        self.vrRestoreButton = qt.QPushButton("恢复原始体积")
        self.vrRestoreButton.setToolTip("移除掩膜体积并恢复原始 MRI 的 3D 显示")
        vrRow.addWidget(self.vrRestoreButton)
        form.addRow(vrRow)

        vrHint = qt.QLabel(
            "说明：Slicer 5.12 的体积渲染只支持盒形（ROI）裁剪，不支持任意形状裁剪，"
            "因此本模块采用“掩膜体积”方案：把通道内体素置为背景值后体积渲染，"
            "通道即显示为透明隧道。勾选“实时更新”后，拖动两点或调整参数会自动刷新掩膜。")
        vrHint.setWordWrap(True)
        vrHint.setStyleSheet("color: gray;")
        form.addRow(vrHint)

        self.vrStatus = qt.QLabel("—")
        self.vrStatus.setWordWrap(True)
        form.addRow(self.vrStatus)

        note = qt.QLabel(
            "说明：本工具为教学与术前规划的定性模拟，用于理解造瘘通道的形态与路径，"
            "不构成精确生物力学预测，不能用于实际手术决策。")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        self.layout.addWidget(note)
        self.layout.addStretch(1)

        # ---- 信号连接 ----
        self.createLineButton.clicked.connect(self._on_create_line)
        self.toggleLineButton.toggled.connect(self._on_toggle_line)
        self.lineSelector.connect("currentNodeChanged(vtkMRMLNode*)",
                                  self._on_line_node_changed)
        self.shapeCombo.currentIndexChanged.connect(self._on_channel_changed)
        self.ra0Slider.valueChanged.connect(self._on_channel_changed)
        self.ra1Slider.valueChanged.connect(self._on_channel_changed)
        self.ratioSlider.valueChanged.connect(self._on_channel_changed)
        self.rotSlider.valueChanged.connect(self._on_channel_changed)
        self.marginSlider.valueChanged.connect(self._on_channel_changed)
        self.capExitCheck.toggled.connect(self._on_channel_changed)
        self.toggleChannelModelButton.toggled.connect(self._on_toggle_channel_model)
        self.vrCreateButton.clicked.connect(self._on_vr_create)
        self.vrRestoreButton.clicked.connect(self._on_vr_restore)

        # 防抖定时器：拖动/调参停止约 250ms 后才刷新掩膜，避免拖动过程反复计算
        self._vrUpdateTimer = qt.QTimer(self.parent)
        self._vrUpdateTimer.setInterval(250)
        self._vrUpdateTimer.setSingleShot(True)
        self._vrUpdateTimer.timeout.connect(self._apply_vr_update)

    def cleanup(self):
        self._remove_line_observer()
        try:
            self._vrUpdateTimer.stop()
        except Exception:
            pass
        self.removeObservers()

    # ---------------- 辅助 ----------------

    def _remove_line_observer(self):
        if self._lineObserverTag is not None and self._observedLineNode is not None:
            self._observedLineNode.RemoveObserver(self._lineObserverTag)
        self._lineObserverTag = None
        self._observedLineNode = None

    def _current_line(self):
        return self.lineSelector.currentNode()

    def _channel_params(self):
        return {
            "shape": self.shapeCombo.currentText,
            "ra0": float(self.ra0Slider.value),
            "ra1": float(self.ra1Slider.value),
            "ratio": float(self.ratioSlider.value),
            "margin": float(self.marginSlider.value),
            "capExit": bool(self.capExitCheck.isChecked()),
            "rotDeg": float(self.rotSlider.value),
        }

    # ---------------- 事件 ----------------

    def _on_create_line(self):
        volume = self.mriSelector.currentNode()
        if volume is None:
            qt.QMessageBox.warning(None, "提示", "请先选择 MRI 体积。")
            return
        try:
            lineNode = self.logic.createChannelLineFromVolume(volume, depth_mm=40.0)
            self.lineSelector.setCurrentNode(lineNode)
            self.toggleLineButton.setChecked(False)
            self.toggleLineButton.setText("隐藏通道线")
            self._on_channel_changed()
        except Exception as e:
            qt.QMessageBox.warning(None, "创建失败", str(e))

    def _on_toggle_line(self, checked):
        lineNode = self._current_line()
        if lineNode is not None:
            lineNode.SetDisplayVisibility(not checked)
        self.toggleLineButton.setText("显示通道线" if checked else "隐藏通道线")

    def _on_toggle_channel_model(self, checked):
        self.logic.setChannelModelVisible(bool(checked))
        self.toggleChannelModelButton.setText(
            "隐藏通道模型" if checked else "显示通道模型")

    def _on_line_node_changed(self):
        """通道线节点切换时重新挂接事件监听。"""
        self._remove_line_observer()
        lineNode = self._current_line()
        if lineNode is not None:
            self._observedLineNode = lineNode
            self._lineObserverTag = lineNode.AddObserver(
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._on_line_modified)
        self._on_channel_changed()

    def _on_line_modified(self, caller, event):
        """两点被拖动时实时更新通道。"""
        self._on_channel_changed()

    def _on_channel_changed(self):
        """任何通道参数变化时立即重建通道示意模型，并按需调度掩膜实时刷新。"""
        lineNode = self._current_line()
        if lineNode is None:
            self.depthLabel.setText("通道深度：-- mm")
            return
        params = self._channel_params()
        try:
            geo = self.logic.currentChannelGeometry(lineNode)
            self.logic.updateChannelModel(
                lineNode, params["shape"], params["ra0"], params["ra1"],
                params["ratio"], params["margin"], params["capExit"],
                params["rotDeg"])
            if geo is not None:
                entry, target, depth = geo
                ra0f, rb0f, ra1f, rb1f = self.logic.shapeRadii(
                    params["shape"], params["ra0"], params["ra1"], params["ratio"])
                vol = FistulaChannelCore.channel_volume(
                    ra0f, rb0f, ra1f, rb1f, depth)
                self.depthLabel.setText(
                    f"通道深度：{depth:.1f} mm　通道体积：约 {vol:.0f} mm³")
        except Exception as e:
            self.depthLabel.setText(f"通道参数错误：{e}")
            return
        if self._vr_active() and self.realtimeCheck.isChecked():
            self._vrUpdateTimer.start()

    def _vr_active(self):
        """体积渲染掩膜是否已生成并仍在场景中。"""
        return (self.logic.maskVolumeNode is not None
                and slicer.mrmlScene.IsNodePresent(self.logic.maskVolumeNode))

    def _apply_vr_update(self):
        """防抖回调：拖动两点/调整参数后自动原地刷新掩膜体积（无需点击）。"""
        if not self._vr_active() or not self.realtimeCheck.isChecked():
            return
        lineNode = self._current_line()
        volume = self.mriSelector.currentNode()
        if lineNode is None or volume is None:
            return
        params = self._channel_params()
        try:
            maskNode, nMasked = self.logic.updateFistulaMaskVolume(
                volume, lineNode, params["shape"], params["ra0"], params["ra1"],
                params["ratio"], params["margin"], params["capExit"],
                params["rotDeg"], backgroundValue=float(self.backgroundSpin.value))
            if maskNode is not None:
                self.vrStatus.setText(f"已实时更新掩膜（通道内 {nMasked} 个体素）。")
        except Exception as e:
            self.vrStatus.setText(f"掩膜更新失败：{e}")

    def _on_vr_create(self):
        lineNode = self._current_line()
        if lineNode is None:
            qt.QMessageBox.information(None, "提示", "请先在①步新建两点通道。")
            return
        volume = self.mriSelector.currentNode()
        if volume is None:
            qt.QMessageBox.warning(None, "体积渲染", "请先选择 MRI 体积。")
            return
        params = self._channel_params()
        try:
            slicer.util.showStatusMessage("正在生成掩膜体积并打开体积渲染…")
            slicer.app.processEvents()
            maskNode, nMasked = self.logic.createFistulaMaskVolume(
                volume, lineNode, params["shape"], params["ra0"], params["ra1"],
                params["ratio"], params["margin"], params["capExit"],
                params["rotDeg"], backgroundValue=float(self.backgroundSpin.value))
            self.logic.setupVolumeRendering(maskNode, volume)
            self.vrStatus.setText(
                f"已生成掩膜体积（通道内 {nMasked} 个体素置为背景值）并打开 VR。\n"
                "3D 视图中通道区域显示为透明隧道；勾选“实时更新”后，"
                "拖动两点或调整参数将自动刷新掩膜，无需再点按钮。")
        except Exception as e:
            self.vrStatus.setText(f"体积渲染失败：{e}")
            qt.QMessageBox.warning(None, "体积渲染失败", str(e))
        finally:
            slicer.util.showStatusMessage("")

    def _on_vr_restore(self):
        maskNode = self.logic.maskVolumeNode
        try:
            # 原始体积由掩膜节点属性自动找回，避免用户切换选择后恢复错对象
            self.logic.restoreVolumeRendering(maskNode, None)
        except Exception as e:
            self.vrStatus.setText(f"恢复失败：{e}")
            qt.QMessageBox.warning(None, "恢复失败", str(e))
            return
        try:
            self._vrUpdateTimer.stop()
        except Exception:
            pass
        self.vrStatus.setText("已恢复：原始体积重新显示，掩膜体积已移除。")
