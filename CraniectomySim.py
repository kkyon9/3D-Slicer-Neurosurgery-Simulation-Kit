# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
"""颅骨切除模拟（CraniectomySim）—— 3D Slicer 脚本模块

功能：在 3D Slicer 中模拟颅骨切除 / 骨窗建立（去骨瓣减压、颅底入路骨质切除等）：
  - 用 Segment 标记待切除颅骨区域（阈值辅助 / 画笔 / 球刷 / 剪刀均可）；
  - 一键执行切除：标记区填充为空气（-1000 HU），生成新 Volume 并自动开启体渲染；
  - 实时预览：边画边切、擦除/撤销自动还原；
  - 一键测量切除部分：体积(mm³/ml)、表面积、OBB 长宽高、Feret 直径、中心坐标，可导出 CSV；
  - 测量后在 3D 视图显示长/宽/高标注线（OBB 三轴）；
  - 一键显示已切除骨质模型（闭合表面）并隐藏其他结构，可一键恢复；
  - 支持多 Segment 并集切除（如左右骨窗 / 多入路联合）。

架构与其他外科模拟插件一致（Widget / Logic 两层 + 自测试）：
  - 不需要安装任何额外依赖（只用 Slicer 自带的 numpy / VTK）；
  - 不会修改原始 Volume，切除结果全部生成在新节点上；
  - 在 Segment Editor 中绘画即可实时预览切除效果。

安装与使用方法见同目录《使用说明.md》。
"""

import os
import csv
import numpy as np
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin


# =============================================================================
# 模块注册信息
# =============================================================================
class CraniectomySim(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "颅骨切除模拟"
        self.parent.categories = ["神经外科模拟手术套件"]
        self.parent.dependencies = []
        self.parent.contributors = ["myz"]
        self.parent.helpText = """
在体渲染（Volume Rendering）场景下模拟颅骨切除：<br>
1. 选择输入 CT Volume 与 Segmentation；<br>
2. 用画笔/球刷/剪刀或"阈值辅助"在 Segment 中标出要切除的颅骨区域；<br>
3. 点击"执行切除并渲染"，生成切除后的新 Volume 并自动体渲染；<br>
4. 点击"测量切除区域"，获得体积、表面积、长宽高（OBB）、Feret 直径等数据，可导出 CSV，并在 3D 视图显示长/宽/高标注线；<br>
5. 点击"一键显示已切除骨质模型"，仅显示切除骨块的 3D 模型，再次点击恢复。
"""
        self.parent.acknowledgementText = "基于 3D Slicer Segmentations / VolumeRendering / SegmentStatistics 模块实现。"


# =============================================================================
# 界面
# =============================================================================
class CraniectomySimWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = CraniectomySimLogic()
        self.outputVolumeNode = None

    # ------------------------------------------------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.layout.setAlignment(qt.Qt.AlignTop)

        # ---- 1. 输入数据 ----
        inputBox = ctk.ctkCollapsibleButton()
        inputBox.text = "1. 输入数据"
        inputForm = qt.QFormLayout(inputBox)

        self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector.selectNodeUponCreation = True
        self.inputVolumeSelector.addEnabled = False
        self.inputVolumeSelector.removeEnabled = False
        self.inputVolumeSelector.noneEnabled = True
        self.inputVolumeSelector.showHidden = False
        self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        inputForm.addRow("输入 Volume (CT):", self.inputVolumeSelector)

        segRow = qt.QHBoxLayout()
        self.segmentationSelector = slicer.qMRMLNodeComboBox()
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.selectNodeUponCreation = True
        self.segmentationSelector.addEnabled = False
        self.segmentationSelector.removeEnabled = False
        self.segmentationSelector.noneEnabled = True
        self.segmentationSelector.showHidden = False
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.newSegmentationButton = qt.QPushButton("新建")
        self.newSegmentationButton.setFixedWidth(60)
        segRow.addWidget(self.segmentationSelector)
        segRow.addWidget(self.newSegmentationButton)
        inputForm.addRow("Segmentation:", segRow)

        segNameRow = qt.QHBoxLayout()
        self.segmentSelector = qt.QComboBox()
        self.segmentSelector.setEditable(True)
        self.segmentSelector.setInsertPolicy(qt.QComboBox.NoInsert)
        self.segmentSelector.setToolTip("选择要切除/测量的 Segment；没有时输入名称并点右侧「新建」")
        self.newSegmentButton = qt.QPushButton("新建")
        self.newSegmentButton.setFixedWidth(60)
        self.newSegmentButton.setToolTip("以左侧输入的名称新建一个 Segment")
        segNameRow.addWidget(self.segmentSelector)
        segNameRow.addWidget(self.newSegmentButton)
        inputForm.addRow("切除区域 Segment:", segNameRow)

        self.layout.addWidget(inputBox)

        # ---- 2. 标记切除区域 ----
        markBox = ctk.ctkCollapsibleButton()
        markBox.text = "2. 标记切除区域"
        markForm = qt.QFormLayout(markBox)

        self.openSegEditorButton = qt.QPushButton("打开 Segment Editor 进行标记")
        self.openSegEditorButton.toolTip = "跳转到 Segment Editor 模块，并自动选中上面的 Segment，用画笔/球刷/剪刀标记要切除的颅骨"
        markForm.addRow(self.openSegEditorButton)

        thrRow = qt.QHBoxLayout()
        self.thrMinSpin = qt.QSpinBox()
        self.thrMinSpin.setRange(-2000, 4000)
        self.thrMinSpin.setValue(300)
        self.thrMaxSpin = qt.QSpinBox()
        self.thrMaxSpin.setRange(-2000, 4000)
        self.thrMaxSpin.setValue(4000)
        thrRow.addWidget(qt.QLabel("HU"))
        thrRow.addWidget(self.thrMinSpin)
        thrRow.addWidget(qt.QLabel("~"))
        thrRow.addWidget(self.thrMaxSpin)
        self.thresholdButton = qt.QPushButton("阈值分割写入 Segment")
        self.thresholdButton.toolTip = "将 HU 范围内的体素（默认 300~4000 为骨组织）写入当前 Segment，之后可用剪刀/擦除精修"
        thrRow.addWidget(self.thresholdButton)
        markForm.addRow("阈值辅助:", thrRow)

        self.layout.addWidget(markBox)

        # ---- 3. 执行切除 ----
        cutBox = ctk.ctkCollapsibleButton()
        cutBox.text = "3. 执行切除"
        cutForm = qt.QFormLayout(cutBox)

        self.fillValueSpin = qt.QSpinBox()
        self.fillValueSpin.setRange(-4000, 4000)
        self.fillValueSpin.setValue(-1000)
        self.fillValueSpin.toolTip = "切除区域在新 Volume 中填充的 HU 值，-1000 为空气（体渲染下不可见）"
        cutForm.addRow("填充值 (HU):", self.fillValueSpin)

        self.hideOriginalCheck = qt.QCheckBox("切除后隐藏原始体渲染")
        self.hideOriginalCheck.setChecked(True)
        cutForm.addRow(self.hideOriginalCheck)

        self.cutAllSegmentsCheck = qt.QCheckBox("合并切除全部 Segments（左右骨窗同时生效）")
        self.cutAllSegmentsCheck.setChecked(False)
        self.cutAllSegmentsCheck.toolTip = ("勾选后，一次性切除会把当前 Segmentation 里所有 Segment 的"
                                            "标记合并到同一个新 Volume 里；不勾选则只切下拉框选中的那一个。")
        cutForm.addRow(self.cutAllSegmentsCheck)

        self.liveCutButton = qt.QPushButton("开启实时切除预览")
        self.liveCutButton.setCheckable(True)
        self.liveCutButton.setStyleSheet("font-weight: bold; padding: 6px;")
        self.liveCutButton.toolTip = ("开启后，在 Segment Editor 里画一笔，体渲染就实时缺一块；"
                                      "擦除/撤销会自动还原。建议先用 Segment Editor 选好画笔再开启。")
        cutForm.addRow(self.liveCutButton)

        self.cutButton = qt.QPushButton("执行切除并渲染（一次性）")
        self.cutButton.setStyleSheet("padding: 6px;")
        cutForm.addRow(self.cutButton)

        self.layout.addWidget(cutBox)

        # ---- 4. 测量 ----
        measureBox = ctk.ctkCollapsibleButton()
        measureBox.text = "4. 测量切除部分"
        measureForm = qt.QVBoxLayout(measureBox)

        btnRow = qt.QHBoxLayout()
        self.measureButton = qt.QPushButton("测量当前 Segment")
        self.measureAllButton = qt.QPushButton("测量全部 Segments")
        self.exportButton = qt.QPushButton("导出 CSV")
        self.exportButton.enabled = False
        btnRow.addWidget(self.measureButton)
        btnRow.addWidget(self.measureAllButton)
        btnRow.addWidget(self.exportButton)
        measureForm.addLayout(btnRow)

        self.showObbCheck = qt.QCheckBox("测量后在 3D 视图显示长/宽/高标注线")
        self.showObbCheck.setChecked(True)
        self.showObbCheck.toolTip = "勾选后，每次测量结束会在 3D 视图显示已切除部分的三条主轴线段（长/宽/高，即 OBB 三轴）"
        measureForm.addWidget(self.showObbCheck)

        self.showBoneModelButton = qt.QPushButton("一键显示已切除骨质模型")
        self.showBoneModelButton.setCheckable(True)
        self.showBoneModelButton.toolTip = ("将当前 Segmentation 中所有 Segment 标记的切除骨质合并为一个闭合表面模型，"
                                            "并隐藏其他所有结构（仅保留该模型在 3D 视图显示）；再次点击恢复全部显示")
        measureForm.addWidget(self.showBoneModelButton)

        self.resultTable = qt.QTableWidget(0, 3)
        self.resultTable.setHorizontalHeaderLabels(["Segment", "指标", "数值"])
        self.resultTable.horizontalHeader().setStretchLastSection(True)
        measureForm.addWidget(self.resultTable)

        self.layout.addWidget(measureBox)

        # ---- 日志 ----
        self.logEdit = qt.QPlainTextEdit()
        self.logEdit.readOnly = True
        self.logEdit.setMaximumHeight(100)
        self.layout.addWidget(self.logEdit)

        self.layout.addStretch(1)

        # ---- 信号 ----
        self.newSegmentationButton.connect("clicked()", self.onNewSegmentation)
        self.newSegmentButton.connect("clicked()", self.onNewSegment)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.segmentSelector.connect("currentIndexChanged(int)", self.onSegmentSelectorChanged)
        self.segmentSelector.lineEdit().connect("editingFinished()", self.onSegmentNameEdited)
        self.openSegEditorButton.connect("clicked()", self.onOpenSegmentEditor)
        self.thresholdButton.connect("clicked()", self.onThresholdSegment)
        self.liveCutButton.connect("toggled(bool)", self.onLiveCutToggled)
        self.cutButton.connect("clicked()", self.onCut)
        self.measureButton.connect("clicked()", self.onMeasure)
        self.measureAllButton.connect("clicked()", self.onMeasureAll)
        self.exportButton.connect("clicked()", self.onExportCSV)
        self.showBoneModelButton.connect("toggled(bool)", self.onShowBoneModelToggled)

        self.refreshSegmentSelector()

        # ---- 实时模式状态 ----
        self._liveActive = False
        self._liveVol = None       # 输入 Volume
        self._liveSegNode = None
        self._liveSegId = None
        # 开启实时前 Segmentation 的显示状态，关闭时恢复
        self._liveOriginalVisibility = True
        self._liveOriginalVisibility3D = True
        self._liveOriginalOpacity3D = 1.0
        self._liveOriginalOpacity2DFill = 1.0
        self._liveOriginalOpacity2DOutline = 1.0
        self._liveOriginalPreferred3D = ""
        # 实时模式下 3D 表面的透明度：
        # Slicer 的 3D 画笔靠拾取 Segment 的 3D 表面网格来定位，而 VTK 拾取器只认
        # "可见且透明度 > 0" 的 actor。因此不能把 3D 表面完全隐藏，否则在 3D 视图里
        # 画第二笔及以后会拾取不到任何东西（第一笔若在 2D 里画则正常）。
        # 设为一个很小的透明度：肉眼几乎看不见、不遮挡体渲染，但 3D 画笔始终可用。
        self._liveSurfaceOpacity3D = 0.1
        self._liveTimer = qt.QTimer()
        self._liveTimer.setSingleShot(True)
        self._liveTimer.setInterval(300)  # 300ms 节流，画笔连续移动时不会每笔都全量刷新
        self._liveTimer.connect("timeout()", self._applyLiveUpdate)
        # 兜底轮询：万一某些版本/路径下画笔事件没有送达（例如 5.10 真实 3D 绘画时
        # SegmentModified 偶发漏发），每秒检查一次 Segment 数据有没有变化，有变化就补一次更新。
        self._liveSafetyTimer = qt.QTimer()
        self._liveSafetyTimer.setInterval(1000)
        self._liveSafetyTimer.connect("timeout()", self._onLiveSafetyCheck)
        self._liveLastSegMTime = 0

        # 测量标注线（长/宽/高）与"已切除骨质模型"视图状态
        self._obbLineNodes = []
        self._boneModelNode = None
        self._visibilitySnapshot = []
        self._boneViewActive = False

    # ------------------------------------------------------------------
    def cleanup(self):
        self._stopLiveCut()
        if getattr(self, "_boneViewActive", False):
            self._restoreVisibility(getattr(self, "_visibilitySnapshot", []))
            self._boneViewActive = False
        ScriptedLoadableModuleWidget.cleanup(self)

    # ------------------------------------------------------------------
    def log(self, msg):
        self.logEdit.appendPlainText(str(msg))
        slicer.app.processEvents()

    def _validateInputs(self, needSegmentation=True):
        vol = self.inputVolumeSelector.currentNode()
        if not vol:
            qt.QMessageBox.warning(self.parent, "提示", "请先选择输入 Volume。")
            return None, None
        seg = self.segmentationSelector.currentNode() if needSegmentation else None
        if needSegmentation and not seg:
            qt.QMessageBox.warning(self.parent, "提示", "请先选择或新建 Segmentation。")
            return None, None
        return vol, seg

    # ------------------------------------------------------------------
    # Segment 下拉框管理
    # ------------------------------------------------------------------
    def _currentSegmentName(self):
        """返回当前 Segment 下拉框中的名称（已脱敏）。"""
        return self.segmentSelector.currentText.strip() or "切除区域"

    def _currentSegmentId(self, segNode):
        """根据当前下拉框名称返回 Segment ID；不存在则返回空字符串。"""
        if not segNode:
            return ""
        return segNode.GetSegmentation().GetSegmentIdBySegmentName(self._currentSegmentName())

    def _ensureSegment(self, segNode, name=None):
        """确保 Segmentation 中存在指定名称的 Segment，返回 segmentID。"""
        name = (name or self._currentSegmentName()).strip() or "切除区域"
        segmentation = segNode.GetSegmentation()
        segId = segmentation.GetSegmentIdBySegmentName(name)
        if not segId:
            # 第一个参数是 segmentId（自动分配），第二个参数才是显示名称
            segId = segmentation.AddEmptySegment("", name)
        self.refreshSegmentSelector(segNode, selectName=name)
        return segId

    def refreshSegmentSelector(self, segNode=None, selectName=None):
        """刷新 Segment 下拉框，列出当前 Segmentation 中所有 Segment 名称。"""
        segNode = segNode or self.segmentationSelector.currentNode()
        self.segmentSelector.blockSignals(True)
        self.segmentSelector.clear()
        names = []
        if segNode:
            segmentation = segNode.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                segId = segmentation.GetNthSegmentID(i)
                names.append(segmentation.GetSegment(segId).GetName())
        if not names:
            names = ["切除区域"]
        self.segmentSelector.addItems(names)
        if selectName and selectName in names:
            self.segmentSelector.setCurrentIndex(names.index(selectName))
        elif "切除区域" in names:
            self.segmentSelector.setCurrentIndex(names.index("切除区域"))
        else:
            self.segmentSelector.setCurrentIndex(0)
        self.segmentSelector.blockSignals(False)

    def onSegmentationNodeChanged(self, node=None):
        self.refreshSegmentSelector(node)

    def onSegmentSelectorChanged(self, index):
        # 用户手动切换 Segment，无需额外动作；名称会由 _currentSegmentName 提供
        pass

    def onSegmentNameEdited(self):
        # 用户在可编辑框里敲了回车/失去焦点：如果该名称不存在，自动新建
        segNode = self.segmentationSelector.currentNode()
        if not segNode:
            return
        name = self._currentSegmentName()
        segId = segNode.GetSegmentation().GetSegmentIdBySegmentName(name)
        if not segId:
            self._ensureSegment(segNode, name)

    def onNewSegment(self):
        segNode = self.segmentationSelector.currentNode()
        if not segNode:
            qt.QMessageBox.warning(self.parent, "提示", "请先选择或新建 Segmentation。")
            return
        name = self._currentSegmentName()
        self._ensureSegment(segNode, name)
        self.log("已新建/选中 Segment: " + name)

    # ------------------------------------------------------------------
    def onNewSegmentation(self):
        vol = self.inputVolumeSelector.currentNode()
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "切除标记")
        if vol:
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(vol)
        segNode.CreateDefaultDisplayNodes()
        # 新建时自动创建一个默认 Segment，避免用户手动再点一次
        self._ensureSegment(segNode, "切除区域")
        self.segmentationSelector.setCurrentNode(segNode)
        self.log("已新建 Segmentation: " + segNode.GetName())

    def _jumpToSegmentEditor(self, vol, segNode, segId):
        """内部用：跳转到 Segment Editor 并选中指定 Segment。"""
        slicer.util.selectModule("SegmentEditor")
        segEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        segEditorWidget.setSegmentationNode(segNode)
        segEditorWidget.setSourceVolumeNode(vol)
        segEditorWidget.setCurrentSegmentID(segId)
        segEditorWidget.setActiveEffectByName("Paint")

    def onOpenSegmentEditor(self):
        vol, segNode = self._validateInputs()
        if not vol:
            return
        segId = self._ensureSegment(segNode)
        self._jumpToSegmentEditor(vol, segNode, segId)
        self.log("已跳转 Segment Editor，当前段: " + self._currentSegmentName())

    def onThresholdSegment(self):
        vol, segNode = self._validateInputs()
        if not vol:
            return
        segId = self._ensureSegment(segNode)
        lo, hi = self.thrMinSpin.value, self.thrMaxSpin.value
        try:
            n = self.logic.thresholdToSegment(vol, segNode, segId, lo, hi)
            self.log("阈值分割完成: %d~%d HU, 共 %d 体素写入 Segment「%s」。" % (lo, hi, n, self._currentSegmentName()))
        except Exception as e:
            qt.QMessageBox.critical(self.parent, "错误", "阈值分割失败:\n" + str(e))

    # ------------------------------------------------------------------
    # 实时切除预览
    # ------------------------------------------------------------------
    def onLiveCutToggled(self, checked):
        if checked:
            vol, segNode = self._validateInputs()
            if not vol:
                self.liveCutButton.setChecked(False)
                return
            segId = self._ensureSegment(segNode)
            try:
                inVR = None
                for i in range(vol.GetNumberOfDisplayNodes()):
                    d = vol.GetNthDisplayNode(i)
                    if d.IsA("vtkMRMLVolumeRenderingDisplayNode"):
                        inVR = d
                        break
                outVol = self.logic.startLiveCut(
                    vol, segNode, segId,
                    fillValue=self.fillValueSpin.value,
                    inputVRDisplayNode=inVR,
                    hideOriginal=self.hideOriginalCheck.checked,
                )
            except Exception as e:
                qt.QMessageBox.critical(self.parent, "错误", "开启实时预览失败:\n" + str(e))
                self.liveCutButton.setChecked(False)
                return

            import vtkSegmentationCorePython as vtkSegmentationCore
            # 保存当前显示状态。实时模式下：
            #  - 3D 表面设为低透明度半透明（保留"可见"，否则 3D 画笔无法拾取表面，见 _liveSurfaceOpacity3D 注释）
            #  - 2D 填充/轮廓完全透明（切片视图里不遮挡）
            #  - 确保 closed surface 表示存在并被 3D 视图使用，画笔在 3D 里才能拾取
            displayNode = segNode.GetDisplayNode()
            if displayNode:
                self._liveOriginalVisibility = bool(displayNode.GetVisibility())
                self._liveOriginalVisibility3D = bool(displayNode.GetVisibility3D())
                self._liveOriginalOpacity3D = float(displayNode.GetOpacity3D())
                self._liveOriginalOpacity2DFill = float(displayNode.GetOpacity2DFill())
                self._liveOriginalOpacity2DOutline = float(displayNode.GetOpacity2DOutline())
                self._liveOriginalPreferred3D = displayNode.GetPreferredDisplayRepresentationName3D() or ""
                closedSurfaceRepName = vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
                displayNode.SetVisibility(True)
                displayNode.SetVisibility3D(True)
                displayNode.SetOpacity3D(self._liveSurfaceOpacity3D)
                displayNode.SetOpacity2DFill(0.0)
                displayNode.SetOpacity2DOutline(0.0)
                displayNode.SetPreferredDisplayRepresentationName3D(closedSurfaceRepName)
                # 提前生成 closed surface：让 3D 视图里立即有可拾取的表面网格
                segNode.GetSegmentation().CreateRepresentation(closedSurfaceRepName)

            self._liveVol = vol
            self._liveSegNode = segNode
            self._liveSegId = segId
            self._liveActive = True
            self.outputVolumeNode = outVol
            self.addObserver(segNode.GetSegmentation(),
                             vtkSegmentationCore.vtkSegmentation.SegmentModified,
                             self._onLiveSegmentModified)
            # 主表示（labelmap）被修改时一定会触发 SourceRepresentationModified，
            # 多监听一个事件，防止 SegmentModified 在个别版本/路径下漏发。
            self.addObserver(segNode.GetSegmentation(),
                             vtkSegmentationCore.vtkSegmentation.SourceRepresentationModified,
                             self._onLiveSegmentModified)
            self._liveLastSegMTime = self._segmentLabelmapMTime(segNode, segId)
            self._liveSafetyTimer.start()
            self.liveCutButton.text = "关闭实时切除预览"
            self.cutButton.enabled = False

            # 自动跳转 Segment Editor 并选中当前 Segment，画笔即开即用
            self._jumpToSegmentEditor(vol, segNode, segId)

            self.log("实时切除预览已开启: %s（Segmentation 2D 已隐藏，3D 表面设为半透明以便画笔拾取；在 Segment Editor 中绘制即实时更新）" % outVol.GetName())
        else:
            self._stopLiveCut()
            self.log("实时切除预览已关闭（输出 Volume 保留，可继续测量）。")

    def _onLiveSegmentModified(self, caller, event, callData=None):
        # 画笔事件可能非常密集，用单发定时器节流：最后一笔结束 300ms 后才更新
        if self._liveActive:
            self._liveTimer.start()

    def _applyLiveUpdate(self):
        if not self._liveActive:
            return
        try:
            segId = self._syncLiveSegmentId()
            changed = self.logic.updateLiveCut(
                self._liveVol, self._liveSegNode, segId,
                fillValue=self.fillValueSpin.value)
            self._liveLastSegMTime = self._segmentLabelmapMTime(self._liveSegNode, segId)
            if changed:
                try:
                    slicer.util.forceRenderAllViews()
                except AttributeError:
                    pass
        except Exception as e:
            import traceback
            self.log("实时更新失败: " + str(e) + "\n" + traceback.format_exc())

    def _onLiveSafetyCheck(self):
        """兜底：事件漏接时，检测到 Segment 数据有变化就补一次更新。"""
        if not self._liveActive or self._liveTimer.isActive():
            return
        try:
            if self._segmentLabelmapMTime(self._liveSegNode, self._liveSegId) != self._liveLastSegMTime:
                self._applyLiveUpdate()
        except Exception:
            pass

    def _segmentLabelmapMTime(self, segNode, segId=None):
        """返回所有 Segment 二值 labelmap 表示的最大修改时间（0 表示还没有 labelmap）。"""
        try:
            import vtkSegmentationCorePython as vtkSegmentationCore
            repName = vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
            segmentation = segNode.GetSegmentation()
            maxMTime = 0
            for i in range(segmentation.GetNumberOfSegments()):
                seg = segmentation.GetSegment(segmentation.GetNthSegmentID(i))
                rep = seg.GetRepresentation(repName) if seg else None
                if rep:
                    maxMTime = max(maxMTime, rep.GetMTime())
            return maxMTime
        except Exception:
            return 0

    def _syncLiveSegmentId(self):
        """实时模式下跟随 Segment Editor 当前选中的 Segment。

        用户在 Segment Editor 里切换/新建 Segment 后，实时预览应对应更新为
        正在画的这个 Segment，而不是开启实时模式时的那一个。
        """
        if not self._liveActive or not self._liveSegNode:
            return self._liveSegId
        try:
            editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            editorSegNode = editor.segmentationNode()
            if editorSegNode and editorSegNode.GetID() == self._liveSegNode.GetID():
                segId = str(editor.currentSegmentID() or "")
                if segId:
                    self._liveSegId = segId
        except Exception:
            # 无界面运行或 Segment Editor 未加载时保持原 Segment
            pass
        return self._liveSegId

    def _stopLiveCut(self):
        if not self._liveActive:
            return
        self._liveActive = False
        self._liveTimer.stop()
        if self._liveSegNode:
            import vtkSegmentationCorePython as vtkSegmentationCore
            self.removeObserver(self._liveSegNode.GetSegmentation(),
                                vtkSegmentationCore.vtkSegmentation.SegmentModified,
                                self._onLiveSegmentModified)
            self.removeObserver(self._liveSegNode.GetSegmentation(),
                                vtkSegmentationCore.vtkSegmentation.SourceRepresentationModified,
                                self._onLiveSegmentModified)
        self.logic.stopLiveCut()
        self._liveSafetyTimer.stop()
        # 恢复 Segmentation 在视图中的原始显示状态
        if self._liveSegNode:
            displayNode = self._liveSegNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(self._liveOriginalVisibility)
                displayNode.SetVisibility3D(self._liveOriginalVisibility3D)
                displayNode.SetOpacity3D(self._liveOriginalOpacity3D)
                displayNode.SetOpacity2DFill(self._liveOriginalOpacity2DFill)
                displayNode.SetOpacity2DOutline(self._liveOriginalOpacity2DOutline)
                displayNode.SetPreferredDisplayRepresentationName3D(self._liveOriginalPreferred3D)
        self._liveVol = None
        self._liveSegNode = None
        self._liveSegId = None
        self._liveOriginalVisibility = True
        self._liveOriginalVisibility3D = True
        self._liveOriginalOpacity3D = 1.0
        self._liveOriginalOpacity2DFill = 1.0
        self._liveOriginalOpacity2DOutline = 1.0
        self._liveOriginalPreferred3D = ""
        self._liveLastSegMTime = 0
        if hasattr(self, "liveCutButton"):
            self.liveCutButton.text = "开启实时切除预览"
            if self.liveCutButton.checked:
                self.liveCutButton.blockSignals(True)
                self.liveCutButton.setChecked(False)
                self.liveCutButton.blockSignals(False)
            self.cutButton.enabled = True

    def onCut(self):
        vol, segNode = self._validateInputs()
        if not vol:
            return
        segmentation = segNode.GetSegmentation()
        if self.cutAllSegmentsCheck.checked:
            # 合并切除全部 Segment：前一个 Segment 的标记也会生效
            segIds = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]
            if not segIds:
                qt.QMessageBox.warning(self.parent, "提示", "当前 Segmentation 中没有任何 Segment。")
                return
        else:
            segId = self._currentSegmentId(segNode)
            if not segId:
                qt.QMessageBox.warning(self.parent, "提示", "Segmentation 中还没有名为「%s」的 Segment，请先标记。" % self._currentSegmentName())
                return
            segIds = [segId]
        try:
            inVR = None
            for i in range(vol.GetNumberOfDisplayNodes()):
                d = vol.GetNthDisplayNode(i)
                if d.IsA("vtkMRMLVolumeRenderingDisplayNode"):
                    inVR = d
                    break
            outVol, removedVoxels = self.logic.performCut(
                vol, segNode, segIds,
                fillValue=self.fillValueSpin.value,
                inputVRDisplayNode=inVR,
                hideOriginal=self.hideOriginalCheck.checked,
            )
            if removedVoxels == 0:
                qt.QMessageBox.warning(self.parent, "提示", "切除区域为空（所选 Segment 均无内容），未生成新 Volume。")
                return
            self.outputVolumeNode = outVol
            self.log("切除完成: %s, 共切除 %d 体素，体渲染已开启。" % (outVol.GetName(), removedVoxels))
        except Exception as e:
            qt.QMessageBox.critical(self.parent, "错误", "切除失败:\n" + str(e))

    def onMeasure(self):
        vol, segNode = self._validateInputs()
        if not vol:
            return
        segId = self._currentSegmentId(segNode)
        if not segId:
            qt.QMessageBox.warning(self.parent, "提示", "找不到待测量的 Segment「%s」。" % self._currentSegmentName())
            return
        try:
            results, obbGeom = self.logic.measureSegment(segNode, segId)
        except Exception as e:
            qt.QMessageBox.critical(self.parent, "错误", "测量失败:\n" + str(e))
            return
        segName = segNode.GetSegmentation().GetSegment(segId).GetName()
        self.lastResults = [(segName, k, v) for k, v in results]
        self._fillResultTable(self.lastResults)
        self.exportButton.enabled = True
        self._updateObbLines([(segName, obbGeom)])
        self.log("测量完成，共 %d 项指标。" % len(results))

    def onMeasureAll(self):
        vol, segNode = self._validateInputs()
        if not vol:
            return
        segmentation = segNode.GetSegmentation()
        if segmentation.GetNumberOfSegments() == 0:
            qt.QMessageBox.warning(self.parent, "提示", "当前 Segmentation 中没有任何 Segment。")
            return
        allRows = []
        obbEntries = []
        try:
            segmentIds = [
                segmentation.GetNthSegmentID(i)
                for i in range(segmentation.GetNumberOfSegments())
            ]
            # Segment Statistics 一次计算整个 Segmentation；避免每个 Segment 重跑完整管线。
            for segId, results, obbGeom in self.logic.measureSegments(segNode, segmentIds):
                segName = segmentation.GetSegment(segId).GetName()
                for k, v in results:
                    allRows.append((segName, k, v))
                obbEntries.append((segName, obbGeom))
        except Exception as e:
            qt.QMessageBox.critical(self.parent, "错误", "测量失败:\n" + str(e))
            return
        self.lastResults = allRows
        self._fillResultTable(allRows)
        self.exportButton.enabled = True
        self._updateObbLines(obbEntries)
        self.log("全部 Segment 测量完成，共 %d 个 Segment，%d 项指标。" % (segmentation.GetNumberOfSegments(), len(allRows)))

    def _fillResultTable(self, rows):
        """rows: [(segmentName, metric, value), ...]"""
        self.resultTable.setRowCount(len(rows))
        for row, (segName, k, v) in enumerate(rows):
            self.resultTable.setItem(row, 0, qt.QTableWidgetItem(segName))
            self.resultTable.setItem(row, 1, qt.QTableWidgetItem(k))
            self.resultTable.setItem(row, 2, qt.QTableWidgetItem(str(v)))

    # ------------------------------------------------------------------
    # 测量标注：在 3D 视图显示已切除部分的长/宽/高（OBB 三轴）线段
    # ------------------------------------------------------------------
    def _removeObbLines(self):
        """删除上一次测量生成的标注线段。"""
        for node in self._obbLineNodes:
            if node and slicer.mrmlScene.GetNodeByID(node.GetID()):
                slicer.mrmlScene.RemoveNode(node)
        self._obbLineNodes = []

    def _updateObbLines(self, entries):
        """entries: [(segmentName, obbGeom 或 None), ...]

        删除旧的标注线，并按最新测量结果重建三条线段：
          长 = OBB 最长轴，宽 = 中间轴，高 = 最短轴；
        线段从 OBB 最小角点出发，沿该轴延伸一个对应直径的长度。
        """
        self._removeObbLines()
        if not self.showObbCheck.checked:
            return
        labels = ["长", "宽", "高"]
        colors = [(0.90, 0.15, 0.15), (0.15, 0.65, 0.15), (0.15, 0.30, 0.95)]
        multi = len(entries) > 1
        for segName, geom in entries:
            if not geom:
                continue
            origin = geom["origin"]
            diameters = geom["diameters"]
            directions = geom["directions"]
            # 按直径降序给三条轴编号：0=长、1=宽、2=高
            order = sorted(range(3), key=lambda i: diameters[i], reverse=True)
            for rank, axisIdx in enumerate(order):
                d = diameters[axisIdx]
                direction = directions[axisIdx]
                p1 = [float(origin[k]) for k in range(3)]
                p2 = [origin[k] + d * direction[k] for k in range(3)]
                name = ("%s-%s" % (segName, labels[rank])) if multi else labels[rank]
                line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
                line.SetName(name)
                line.CreateDefaultDisplayNodes()
                try:
                    line.AddControlPoint(vtk.vtkVector3d(*p1))
                    line.AddControlPoint(vtk.vtkVector3d(*p2))
                except Exception:
                    # 兼容旧版 API：先加两个控制点再逐个定位
                    line.AddNControlPoints(2)
                    line.SetNthControlPointPosition(0, *p1)
                    line.SetNthControlPointPosition(1, *p2)
                displayNode = line.GetDisplayNode()
                if displayNode:
                    displayNode.SetColor(*colors[rank])
                    displayNode.SetVisibility(True)
                    for setterName, value in (("SetVisibility3D", True), ("SetVisibility2D", False)):
                        setter = getattr(displayNode, setterName, None)
                        if setter:
                            try:
                                setter(value)
                            except Exception:
                                pass
                self._obbLineNodes.append(line)
        if self._obbLineNodes:
            self.log("已在 3D 视图显示 %d 条长/宽/高标注线。" % len(self._obbLineNodes))

    # ------------------------------------------------------------------
    # 一键显示已切除骨质模型（隐藏其他所有结构，可一键恢复）
    # ------------------------------------------------------------------
    def onShowBoneModelToggled(self, checked):
        if checked:
            vol, segNode = self._validateInputs()
            if not vol:
                self.showBoneModelButton.blockSignals(True)
                self.showBoneModelButton.setChecked(False)
                self.showBoneModelButton.blockSignals(False)
                return
            if self._liveActive:
                qt.QMessageBox.warning(self.parent, "提示",
                                       "请先关闭实时切除预览，再使用「一键显示已切除骨质模型」。")
                self.showBoneModelButton.blockSignals(True)
                self.showBoneModelButton.setChecked(False)
                self.showBoneModelButton.blockSignals(False)
                return
            try:
                modelNode = self.logic.createRemovedBoneModel(segNode)
            except Exception as e:
                qt.QMessageBox.critical(self.parent, "错误", "生成已切除骨质模型失败:\n" + str(e))
                self.showBoneModelButton.blockSignals(True)
                self.showBoneModelButton.setChecked(False)
                self.showBoneModelButton.blockSignals(False)
                return
            # 复用已有模型节点：避免反复点击生成多个模型
            if self._boneModelNode is not None and slicer.mrmlScene.GetNodeByID(self._boneModelNode.GetID()):
                self._boneModelNode.SetAndObservePolyData(modelNode.GetPolyData())
                displayNode = self._boneModelNode.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(True)
                slicer.mrmlScene.RemoveNode(modelNode)
            else:
                self._boneModelNode = modelNode
            # 保存其他结构的可见性并全部隐藏（保留 2D 切片与当前模型）
            self._visibilitySnapshot = self._hideAllStructures(
                excludeNodeIds=[self._boneModelNode.GetID()])
            displayNode = self._boneModelNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(True)
                setter = getattr(displayNode, "SetVisibility3D", None)
                if setter:
                    try:
                        setter(True)
                    except Exception:
                        pass
            self._boneViewActive = True
            self.log("已显示已切除骨质模型「%s」，其他结构已隐藏；再次点击按钮可恢复显示。" % self._boneModelNode.GetName())
        else:
            self._restoreVisibility(self._visibilitySnapshot)
            self._visibilitySnapshot = []
            if self._boneModelNode is not None and slicer.mrmlScene.GetNodeByID(self._boneModelNode.GetID()):
                displayNode = self._boneModelNode.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)
            self._boneViewActive = False
            self.log("已恢复全部结构显示。")

    def _saveAndHideDisplayNode(self, displayNode):
        """记录显示节点当前的可见性标志并全部隐藏，返回恢复所需信息。"""
        record = {"displayNode": displayNode, "flags": {}}
        for attr in ("Visibility", "Visibility3D", "Visibility2D"):
            getter = getattr(displayNode, "Get" + attr, None)
            setter = getattr(displayNode, "Set" + attr, None)
            if getter is None or setter is None:
                continue
            try:
                record["flags"][attr] = getter()
                setter(False)
            except Exception:
                pass
        return record

    def _hideAllStructures(self, excludeNodeIds=()):
        """隐藏除排除节点外所有可显示结构（模型/分割/标注/体渲染），返回恢复信息列表。"""
        saved = []
        seen = set()
        scene = slicer.mrmlScene
        for i in range(scene.GetNumberOfNodes()):
            node = scene.GetNthNode(i)
            if node is None or node.GetID() in excludeNodeIds:
                continue
            if node.IsA("vtkMRMLDisplayableNode"):
                displayNodes = []
                getDisplayNode = getattr(node, "GetDisplayNode", None)
                if getDisplayNode:
                    d = getDisplayNode()
                    if d:
                        displayNodes.append(d)
                getNth = getattr(node, "GetNthDisplayNode", None)
                if getNth:
                    for j in range(node.GetNumberOfDisplayNodes()):
                        d = getNth(j)
                        if d:
                            displayNodes.append(d)
                if node.IsA("vtkMRMLVolumeNode"):
                    # 2D 切片显示保留，只隐藏体渲染
                    displayNodes = [d for d in displayNodes
                                    if d and d.IsA("vtkMRMLVolumeRenderingDisplayNode")]
                for d in displayNodes:
                    if d and d.GetID() not in seen:
                        seen.add(d.GetID())
                        saved.append(self._saveAndHideDisplayNode(d))
            elif node.IsA("vtkMRMLVolumeRenderingDisplayNode"):
                if node.GetID() not in seen:
                    seen.add(node.GetID())
                    saved.append(self._saveAndHideDisplayNode(node))
        return saved

    def _restoreVisibility(self, saved):
        for record in saved:
            displayNode = record["displayNode"]
            if not displayNode or not slicer.mrmlScene.GetNodeByID(displayNode.GetID()):
                continue
            for attr, value in record["flags"].items():
                setter = getattr(displayNode, "Set" + attr, None)
                if setter:
                    try:
                        setter(value)
                    except Exception:
                        pass

    def onExportCSV(self):
        if not getattr(self, "lastResults", None):
            return
        path = qt.QFileDialog.getSaveFileName(self.parent, "导出测量结果", "切除测量.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Segment", "指标", "数值"])
            writer.writerows(self.lastResults)
        self.log("已导出: " + path)


# =============================================================================
# 逻辑层
# =============================================================================
class CraniectomySimLogic(ScriptedLoadableModuleLogic):

    def thresholdToSegment(self, volumeNode, segNode, segmentId, huMin, huMax):
        """按 HU 阈值生成掩膜并写入指定 Segment，返回体素数。"""
        import vtkSegmentationCorePython as vtkSegmentationCore
        arr = slicer.util.arrayFromVolume(volumeNode)
        # Slicer 要求 Segment 的 labelmap 体素值等于该 Segment 的 label 值
        # （第 1 个=1，第 2 个=2，...），否则多 Segment 导出/合并时会被跳过。
        labelValue = int(segNode.GetSegmentation().GetSegment(segmentId).GetLabelValue())
        maskBool = np.greater_equal(arr, huMin)
        np.logical_and(maskBool, arr <= huMax, out=maskBool)
        voxelCount = int(maskBool.sum())
        if voxelCount == 0:
            return 0
        if labelValue <= np.iinfo(np.uint8).max:
            labelDtype = np.uint8
        elif labelValue <= np.iinfo(np.uint16).max:
            labelDtype = np.uint16
        else:
            labelDtype = np.uint32
        mask = maskBool.astype(labelDtype)
        if labelValue != 1:
            mask *= labelValue

        # numpy -> vtkOrientedImageData（几何与输入 Volume 一致）
        lblNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__tmp_threshold")
        ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRAS)
        lblNode.SetIJKToRASMatrix(ijkToRAS)
        slicer.util.updateVolumeFromArray(lblNode, mask)

        modifierImage = slicer.vtkOrientedImageData()
        modifierImage.DeepCopy(lblNode.GetImageData())
        modifierImage.SetGeometryFromImageToWorldMatrix(ijkToRAS)
        slicer.mrmlScene.RemoveNode(lblNode)

        # 直接写入 Segment 的二进制 labelmap 表示层（不依赖界面类，可无界面运行）
        segment = segNode.GetSegmentation().GetSegment(segmentId)
        repName = vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        segment.AddRepresentation(repName, modifierImage)
        return voxelCount

    def performCut(self, volumeNode, segNode, segmentIds, fillValue=-1000,
                   inputVRDisplayNode=None, hideOriginal=True):
        """
        将 Segment（可多个）标记区域从 Volume 中"切除"：生成新 Volume，切除区填充为背景值。
        segmentIds 可以是单个 Segment ID（str）或 Segment ID 列表（list[str]）。
        自动创建体渲染显示节点，并从输入 Volume 复制传输函数。
        返回 (outputVolumeNode, removedVoxelCount)。
        """
        if isinstance(segmentIds, str):
            segmentIds = [segmentIds]
        # Segment -> labelmap（与输入 Volume 几何一致）
        lblNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__tmp_cutmask")
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segNode, segmentIds, lblNode, volumeNode)
        lbl = slicer.util.arrayFromVolume(lblNode)
        cutMask = lbl > 0
        removedVoxels = int(np.count_nonzero(cutMask))

        if removedVoxels > 0:
            arr = slicer.util.arrayFromVolume(volumeNode).copy()
            arr[cutMask] = fillValue

            outVol = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", volumeNode.GetName() + "_切除后")
            ijkToRAS = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRAS)
            outVol.SetIJKToRASMatrix(ijkToRAS)
            slicer.util.updateVolumeFromArray(outVol, arr)

            # 体渲染
            vrLogic = slicer.modules.volumerendering.logic()
            vrDisp = vrLogic.CreateDefaultVolumeRenderingNodes(outVol)
            if inputVRDisplayNode and inputVRDisplayNode.GetVolumePropertyNode():
                # 复制原 Volume 的传输函数，保证视觉风格一致
                vrDisp.GetVolumePropertyNode().Copy(inputVRDisplayNode.GetVolumePropertyNode())
            vrDisp.SetVisibility(True)
            if hideOriginal and inputVRDisplayNode:
                inputVRDisplayNode.SetVisibility(False)

            # 在切片视图中显示新 Volume（无界面运行时跳过视图操作）
            try:
                slicer.util.setSliceViewerLayers(background=outVol, fit=True)
                slicer.util.resetThreeDViews()
            except AttributeError:
                pass
        else:
            outVol = None

        slicer.mrmlScene.RemoveNode(lblNode)
        return outVol, removedVoxels

    # ------------------------------------------------------------------
    # 实时切除预览：Segment 每次修改后增量更新输出 Volume，
    # 体渲染检测到体数据 Modified 即自动重渲；擦除/撤销可自动还原。
    # ------------------------------------------------------------------
    def startLiveCut(self, volumeNode, segNode, segmentId, fillValue=-1000,
                     inputVRDisplayNode=None, hideOriginal=True):
        """创建实时预览输出 Volume（输入的完整副本）并建立增量更新状态，返回输出 Volume。"""
        outVol = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", volumeNode.GetName() + "_实时切除")
        ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRAS)
        outVol.SetIJKToRASMatrix(ijkToRAS)
        slicer.util.updateVolumeFromArray(outVol, slicer.util.arrayFromVolume(volumeNode).copy())

        vrLogic = slicer.modules.volumerendering.logic()
        vrDisp = vrLogic.CreateDefaultVolumeRenderingNodes(outVol)
        if inputVRDisplayNode and inputVRDisplayNode.GetVolumePropertyNode():
            vrDisp.GetVolumePropertyNode().Copy(inputVRDisplayNode.GetVolumePropertyNode())
        vrDisp.SetVisibility(True)
        if hideOriginal and inputVRDisplayNode:
            inputVRDisplayNode.SetVisibility(False)

        try:
            slicer.util.setSliceViewerLayers(background=outVol, fit=True)
            slicer.util.resetThreeDViews()
        except AttributeError:
            pass  # 无界面运行

        # 增量更新状态：复用的临时 labelmap + 上一帧掩膜缓存
        self._liveLabelmapNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "__tmp_livemask")
        inputArray = slicer.util.arrayFromVolume(volumeNode)
        self._livePrevMask = np.zeros(inputArray.shape, dtype=bool)
        self._liveOutVol = outVol
        self._liveInputArray = inputArray
        self._liveOutputArray = slicer.util.arrayFromVolume(outVol)

        # 若 Segment 已有内容，先应用一次完整掩膜
        self.updateLiveCut(volumeNode, segNode, segmentId, fillValue)
        return outVol

    def updateLiveCut(self, volumeNode, segNode, segmentId, fillValue=-1000):
        """
        对比"所有 Segment 并集"掩膜与上一帧缓存，只更新变化的体素：
          新增标记 -> 填充背景值；取消标记(擦除/撤销) -> 还原原始 HU。
        注意：以全部 Segment 的并集为准，因此切换/新建 Segment 后，
        之前 Segment 的切除区域会继续保留（前一个 Segment 也生效）。
        segmentId 参数保留仅用于兼容旧调用，实际导出全部 Segment。
        返回变化的体素数。
        """
        if getattr(self, "_liveOutVol", None) is None:
            return 0
        # 显式导出全部 Segment（而不是空列表），保证多 Segment 时并集完整
        segmentation = segNode.GetSegmentation()
        allSegIds = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]
        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segNode, allSegIds, self._liveLabelmapNode, volumeNode)
        lbl = slicer.util.arrayFromVolume(self._liveLabelmapNode)
        cur = lbl > 0
        prev = self._livePrevMask

        # 只保留一个变化索引数组，避免为整幅体积创建 add/del 两个额外布尔数组。
        changedIdx = np.flatnonzero(cur.ravel() != prev.ravel())
        changed = int(changedIdx.size)
        if changed == 0:
            return 0

        outFlat = self._liveOutputArray.ravel()
        inFlat = self._liveInputArray.ravel()
        curFlat = cur.ravel()
        prevFlat = prev.ravel()
        changedState = curFlat[changedIdx]
        outFlat[changedIdx] = np.where(
            changedState, fillValue, inFlat[changedIdx])
        prevFlat[changedIdx] = changedState

        self._liveOutVol.GetImageData().Modified()
        self._liveOutVol.Modified()
        # 保险：主动刷新输出 Volume 的体渲染显示节点，
        # 避免个别版本/环境下体渲染没有及时响应 ImageDataModified 事件。
        try:
            for i in range(self._liveOutVol.GetNumberOfDisplayNodes()):
                dispNode = self._liveOutVol.GetNthDisplayNode(i)
                if dispNode and dispNode.IsA("vtkMRMLVolumeRenderingDisplayNode"):
                    dispNode.Modified()
        except Exception:
            pass
        return changed

    def stopLiveCut(self):
        """释放实时模式状态（输出 Volume 保留在场景中）。"""
        if getattr(self, "_liveLabelmapNode", None):
            slicer.mrmlScene.RemoveNode(self._liveLabelmapNode)
        self._liveLabelmapNode = None
        self._livePrevMask = None
        self._liveOutVol = None
        self._liveInputArray = None
        self._liveOutputArray = None

    @staticmethod
    def _computeSegmentStatistics(segNode):
        """执行一次 Segment Statistics 并返回包含全部 Segment 的统计字典。"""
        import SegmentStatistics
        ssLogic = SegmentStatistics.SegmentStatisticsLogic()
        paramNode = ssLogic.getParameterNode()
        paramNode.SetParameter("Segmentation", segNode.GetID())
        # Slicer 5.x 中 OBB / Feret / 表面积等形状指标默认不计算，需显式开启
        for key in ["obb_diameter_mm", "obb_origin_ras", "obb_direction_ras_x",
                    "obb_direction_ras_y", "obb_direction_ras_z",
                    "feret_diameter_mm", "centroid_ras", "surface_area_mm2"]:
            paramNode.SetParameter("LabelmapSegmentStatisticsPlugin." + key + ".enabled", "True")
        ssLogic.computeStatistics()
        return ssLogic.getStatistics()  # 注意：tuple 键 (segmentID, key)

    @staticmethod
    def _formatSegmentStatistics(stats, segmentId):
        """把一个 Segment 的统计字典格式化为界面结果和 OBB 几何。"""

        def getScalar(key, fmt="{:.2f}"):
            v = stats.get((segmentId, key), None)
            if v is None:
                return "N/A"
            try:
                return fmt.format(float(v))
            except (TypeError, ValueError):
                return str(v)

        def getVec(key):
            v = stats.get((segmentId, key), None)
            if v is None:
                return None
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                return None

        volumeMm3 = float(stats.get((segmentId, "LabelmapSegmentStatisticsPlugin.volume_mm3"), 0) or 0)
        obb = getVec("LabelmapSegmentStatisticsPlugin.obb_diameter_mm")
        centroid = getVec("LabelmapSegmentStatisticsPlugin.centroid_ras")
        obbOrigin = getVec("LabelmapSegmentStatisticsPlugin.obb_origin_ras")
        obbDirs = [
            getVec("LabelmapSegmentStatisticsPlugin.obb_direction_ras_x"),
            getVec("LabelmapSegmentStatisticsPlugin.obb_direction_ras_y"),
            getVec("LabelmapSegmentStatisticsPlugin.obb_direction_ras_z"),
        ]
        obbGeom = None
        if (obb and len(obb) == 3 and obbOrigin and len(obbOrigin) == 3
                and all(d and len(d) == 3 for d in obbDirs)):
            obbGeom = {"origin": obbOrigin, "diameters": obb, "directions": obbDirs}

        results = [
            ("体积 (mm³)", "{:.2f}".format(volumeMm3)),
            ("体积 (ml)", "{:.3f}".format(volumeMm3 / 1000.0)),
            ("体素数", stats.get((segmentId, "LabelmapSegmentStatisticsPlugin.voxel_count"), "N/A")),
            ("表面积 (mm²)", getScalar("LabelmapSegmentStatisticsPlugin.surface_area_mm2")),
        ]
        if obb and len(obb) == 3:
            obbSorted = sorted(obb)
            results += [
                ("长 (mm)", "{:.2f}".format(obbSorted[2])),
                ("宽 (mm)", "{:.2f}".format(obbSorted[1])),
                ("高 (mm)", "{:.2f}".format(obbSorted[0])),
                ("OBB 三轴原始值 (mm)", "{:.2f} × {:.2f} × {:.2f}".format(*obb)),
            ]
        results.append(("Feret 最大径 (mm)", getScalar("LabelmapSegmentStatisticsPlugin.feret_diameter_mm")))
        if centroid and len(centroid) == 3:
            results.append(("中心坐标 RAS (mm)", "R {:.2f}, A {:.2f}, S {:.2f}".format(*centroid)))
        return results, obbGeom

    def measureSegments(self, segNode, segmentIds=None):
        """一次统计多个 Segment，返回 [(segmentId, results, obbGeom), ...]。"""
        segmentation = segNode.GetSegmentation()
        if segmentIds is None:
            segmentIds = [
                segmentation.GetNthSegmentID(i)
                for i in range(segmentation.GetNumberOfSegments())
            ]
        stats = self._computeSegmentStatistics(segNode)
        return [
            (segmentId, *self._formatSegmentStatistics(stats, segmentId))
            for segmentId in segmentIds
        ]

    def measureSegment(self, segNode, segmentId):
        """测量单个 Segment；多 Segment 场景请使用 measureSegments 避免重复计算。"""
        _, results, obbGeom = self.measureSegments(segNode, [segmentId])[0]
        return results, obbGeom

    def createRemovedBoneModel(self, segNode):
        """将 Segmentation 中所有 Segment 标记的区域合并为闭合表面模型（已切除骨质）。

        返回 vtkMRMLModelNode；Segment 为空或全部无内容时抛出 ValueError。
        注意：只提取网格，不会修改原始 Volume / Segmentation。
        """
        import vtkSegmentationCorePython as vtkSegmentationCore
        segmentation = segNode.GetSegmentation()
        if segmentation.GetNumberOfSegments() == 0:
            raise ValueError("当前 Segmentation 中没有任何 Segment。")
        # 确保闭合表面表示存在（仅用于提取网格，不影响视图显示）
        closedSurfRepName = vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        segmentation.CreateRepresentation(closedSurfRepName)

        append = vtk.vtkAppendPolyData()
        for i in range(segmentation.GetNumberOfSegments()):
            seg = segmentation.GetSegment(segmentation.GetNthSegmentID(i))
            rep = seg.GetRepresentation(closedSurfRepName) if seg else None
            if rep and rep.GetNumberOfPoints() > 0:
                append.AddInputData(rep)
        append.Update()
        polydata = append.GetOutput()
        if not polydata or polydata.GetNumberOfPoints() == 0:
            raise ValueError("所选 Segment 均无内容，无法生成已切除骨质模型。")

        modelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "已切除骨质模型")
        modelNode.SetAndObservePolyData(polydata)
        modelNode.CreateDefaultDisplayNodes()
        displayNode = modelNode.GetDisplayNode()
        if displayNode:
            displayNode.SetColor(0.92, 0.72, 0.35)
            displayNode.SetVisibility(True)
            setter = getattr(displayNode, "SetVisibility3D", None)
            if setter:
                try:
                    setter(True)
                except Exception:
                    pass
        return modelNode


# =============================================================================
# 自测试
# =============================================================================
class CraniectomySimTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_CutPipeline()

    def test_CutPipeline(self):
        import numpy as np
        # 构造模拟 CT：软组织球 + 骨壳
        vol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "TestCT")
        arr = np.full((60, 60, 60), -1000, dtype=np.int16)
        zz, yy, xx = np.mgrid[0:60, 0:60, 0:60]
        r = np.sqrt((xx - 30.0) ** 2 + (yy - 30.0) ** 2)
        arr[r < 24] = 50
        arr[(r >= 22) & (r < 25)] = 800
        m = vtk.vtkMatrix4x4()
        vol.SetIJKToRASMatrix(m)
        slicer.util.updateVolumeFromArray(vol, arr)

        # 标记切除区域
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "TestSeg")
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(vol)
        segNode.CreateDefaultDisplayNodes()
        cut = np.zeros((60, 60, 60), np.uint8)
        cut[20:40, 35:50, 35:50] = 1
        lblNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(lblNode, cut)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lblNode, segNode)
        slicer.mrmlScene.RemoveNode(lblNode)
        segId = segNode.GetSegmentation().GetNthSegmentID(0)

        # 执行切除
        logic = CraniectomySimLogic()
        outVol, removed = logic.performCut(vol, segNode, segId, fillValue=-1000)
        self.assertIsNotNone(outVol)
        self.assertEqual(removed, 20 * 15 * 15)
        check = slicer.util.arrayFromVolume(outVol)
        self.assertTrue((check[20:40, 35:50, 35:50] == -1000).all())
        self.assertEqual(check[10, 10, 10], arr[10, 10, 10])

        # 测量
        results, obbGeom = logic.measureSegment(segNode, segId)
        results = dict(results)
        self.assertAlmostEqual(float(results["体积 (mm³)"]), 20 * 15 * 15, delta=1.0)
        self.assertTrue(float(results["Feret 最大径 (mm)"]) > 0)
        self.assertIsNotNone(obbGeom)
        self.assertEqual(len(obbGeom["diameters"]), 3)
        self.assertAlmostEqual(sorted(obbGeom["diameters"])[-1], 20.0, delta=2.0)

        # 已切除骨质模型
        modelNode = logic.createRemovedBoneModel(segNode)
        self.assertIsNotNone(modelNode)
        self.assertGreater(modelNode.GetPolyData().GetNumberOfPoints(), 0)
        self.assertGreater(modelNode.GetPolyData().GetNumberOfCells(), 0)

        self.delayDisplay("CraniectomySim 自测试通过")
