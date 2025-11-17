# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools

# Inspect information on Python objects like modules
import inspect
import numpy as np
import warnings
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

# Import the elementwise binary operation module to extract names of all
# specializations (which require PE parallelism to be configured)
import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.fpgadataflow import is_hls_node, is_rtl_node


def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


def common_divisors(numbers):
    separate_divisors = []
    for num in numbers:
        individual_divisors = list(divisors(num))
        separate_divisors.append(individual_divisors)

    return functools.reduce(np.intersect1d, separate_divisors)


# Find the op-type names for all HLS specializations of elementwise binary
# operations
ELEMENTWISE_BINARY_OPS = [
    op_type
    for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
    if issubclass(cls, elementwise_binary_hls.ElementwiseBinaryOperation_hls)
]


class SetFolding(Transformation):
    """Attempt to set parallelism attributes in all nodes to meet a specific
    target expressed as cycles per frame target_cycles_per_frame. For each
    HLSCustomOp node type, the attribute may vary but is typically one of {PE, SIMD},
    and has a certain allowed-maximum value and divisibility constraints,
    which SetFolding will take into account. Note that the algorithm implemented
    by SetFolding is very simple and it is often possible to hand-tune the returned
    parallelism configuration for better results.

    In the returned model, each node's
    cycles_estimate attribute will be set to its estimated number of cycles.

    If two_pass_relaxation is enabled,
    SetFolding will internally run a second time if the target cycles from the
    first pass could not be achieved, instead using the achievable target (which
    may be constrained by a single node) to obtain a balanced pipeline.

    Notable exceptions and special behavior:

    When folding dense convolution/FC compute engines ("MVAU"/MatrixVectorActivation),
    which have two attributes (PE and SIMD):

    * first increases SIMD while weight stream width per PE is <= mvau_wwidth_max
      (configurable in the SetFolding initializer, defaults to 36)
    * then increases PE until the target is met or max PE reached

    When folding depthwise convolutions ("VVAU"/VectorVectorActivation)
    or spatial reduction ops (Pool_Batch):

    * the producer of the node is expected to be a ConvolutionInputGenerator
      with depthwise=1, whose SIMD value will be set equal to the PE value of
      its consumer node
    * the VVAU also supports SIMD ("input window") parallelism next to
      PE ("channels"), but current ConvInpGen limitations require PE to be fully
      unfolded before SIMD is increased
    """

    def __init__(self, target_cycles_per_frame=1000, mvau_wwidth_max=36, two_pass_relaxation=True):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.target_cycles_per_frame:
                # finish if target met
                break

    def apply(self, model):
        graph = model.graph
        # these ops use PE parallelism, up to a max value of NumChannels
        pe_ops = [
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
            *ELEMENTWISE_BINARY_OPS,
        ]
        # these ops use SIMD parallelism, up to a max value of NumChannels
        # ConvolutionInputGenerator has a special case when depthwise=1
        # ConvolutionInputGenerator_rtl supports additional parallelism by
        # setting parallel_window=1 mode after maxing out SIMD
        simd_ops = [
            "FMPadding_rtl",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_rtl",
            "StreamingSplit_hls",
            "StreamingConcat_hls",
        ]
        # these ops are preceded by depthwise SWG and have special behavior,
        # as explained in the SetFolding docstring
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]
        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)
            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")
                node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)
                # increase SIMD until either we meet
                # the target or weight stream becomes
                # too wide
                for simd_val in divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame:
                        # finish if target met
                        break
                    if (
                        node_inst.get_input_datatype(1).bitwidth() * node_inst.get_nodeattr("SIMD")
                        > self.mvau_wwidth_max
                    ):
                        # revert if we've gone above width threshold
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break
                # increase PE until target met or reached max_pe
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in pe_ops:
                # Note: Keep original behavior for all custom-ops defining the
                # NumChannels attribute as it is
                try:
                    max_pe = node_inst.get_nodeattr("NumChannels")
                # Note: Some of the recent additions do not define the
                # NumChannels attribute
                except AttributeError:
                    # We can extract the channels from the normal, i.e., not
                    # folded, shape of the input in these cases
                    max_pe = node_inst.get_normal_input_shape()[-1]
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "LabelSelect_hls":
                max_pe = node_inst.get_nodeattr("Labels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type in depthwise_op_exceptions:
                # init/reset SIMD of VVAU
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    node_inst.set_nodeattr("SIMD", 1)
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # increase SIMD for VVAU once PE is exhausted
                pe = node_inst.get_nodeattr("PE")
                cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                    and cyc > self.target_cycles_per_frame
                ):
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                    swu_node_inst = getCustomOp(swu_node)
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    # enable parallel_window mode of RTL SWG if needed
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        if op_type.startswith("VVAU") and node_inst.get_nodeattr("SIMD") > 1:
                            swu_node_inst.set_nodeattr("parallel_window", 1)
                        else:
                            swu_node_inst.set_nodeattr("parallel_window", 0)
                else:
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        ksize = np.prod(node_inst.get_nodeattr("Kernel"))
                    elif op_type == "Pool_hls":
                        ksize = node_inst.get_nodeattr("KernelSize")
                    else:
                        raise Exception("Undefined edge case for %s" % op_type)
                    if ksize != 1:  # pointwise vvau/pool lack a SWU
                        raise Exception("Expected SWU on DW op input, found " + swu_node.op_type)
            elif op_type in simd_ops:
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                        # init/reset parallel_window mode of RTL SWG
                        if op_type == "ConvolutionInputGenerator_rtl":
                            node_inst.set_nodeattr("parallel_window", 0)
                        self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                        # enable parallel_window mode of RTL SWG if needed
                        simd = node_inst.get_nodeattr("SIMD")
                        cyc = node_inst.get_exp_cycles()
                        if (
                            op_type == "ConvolutionInputGenerator_rtl"
                            and simd == max_simd
                            and cyc > self.target_cycles_per_frame
                        ):
                            node_inst.set_nodeattr("parallel_window", 1)
                    else:
                        # depthwise SWGs are handled separately
                        continue
                elif op_type == "StreamingConcat_hls" or op_type == "StreamingSplit_hls":
                    node_inst.set_nodeattr("SIMD", 1)
                    channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
                    for simd_val in common_divisors(channels_per_stream):
                        node_inst.set_nodeattr("SIMD", simd_val)
                        cyc = node_inst.get_exp_cycles()
                        if cyc < self.target_cycles_per_frame:
                            break
                else:
                    max_simd = node_inst.get_nodeattr("NumChannels")
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
            else:
                warnings.warn("SetFolding doesn't know how to handle op_type " + op_type)

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        if self.two_pass_relaxation:
            perf_dict = model.analysis(dataflow_performance)
            if perf_dict["max_cycles"] > self.target_cycles_per_frame:
                # run again, but with lower target (that we managed) -- this
                # may be coming from a single node's constraints, but we want
                # to balance the entire dataflow pipeline instead
                # no two_pass_relaxation this time -- no guarantee we'll
                # converge otherwise
                warnings.warn(
                    "Node %s is bottleneck with %d cycles, running second pass"
                    % (perf_dict["max_cycles_node_name"], perf_dict["max_cycles"])
                )
                model = model.transform(
                    SetFolding(
                        target_cycles_per_frame=perf_dict["max_cycles"],
                        mvau_wwidth_max=self.mvau_wwidth_max,
                        two_pass_relaxation=False,
                    )
                )

        return (model, False)
    
# ------------------------------------------------------------------------------#
# Name: AnnotateMVAUSparsity
# Author: Changhong Li
# Date: Nov, 2025
# Description: Analyze and annotate MVAU nodes with sparsity information
# ------------------------------------------------------------------------------#
import numpy as np
from onnx import helper
# from finn.transformation.base import Transformation


class AnnotateMVAUSparsity(Transformation):
    """在图中找到所有 MVAU_hls 节点，对其权重(输入2)做稀疏度分析并写回到节点属性里"""

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[AnnotateMVAUSparsity] start")
        print(f"[AnnotateMVAUSparsity] total nodes: {len(graph.node)}")

        for idx, node in enumerate(graph.node):
            # 只处理 MVAU_hls
            if node.op_type != "MVAU_hls":
                continue

            print(f"\n[AnnotateMVAUSparsity] -> processing node #{idx}: name={node.name}, op_type={node.op_type}")

            # MVAU 通常至少有 3 个输入: data, ... , weights
            if len(node.input) < 3:
                print(f"[AnnotateMVAUSparsity]    skip: node has only {len(node.input)} inputs, no weight at index 2")
                continue

            weight_name = node.input[1]
            print(f"[AnnotateMVAUSparsity]    weight input name: {weight_name}")

            # FINN 的 ModelWrapper 通常有 get_initializer
            weight_arr = model.get_initializer(weight_name)
            if weight_arr is None:
                print(f"[AnnotateMVAUSparsity]    skip: weight '{weight_name}' is not an initializer (maybe runtime)")
                continue

            # 转成 np.array 并打印shape
            w_np = weight_arr
            if not isinstance(w_np, np.ndarray):
                w_np = np.array(weight_arr)

            print(f"[AnnotateMVAUSparsity]    weight shape: {w_np.shape}")

            # 计算稀疏度: #zeros / #elements
            flat_w = w_np.flatten()
            total = flat_w.size
            if total == 0:
                print("[AnnotateMVAUSparsity]    warning: weight has 0 elements, set sparsity=0.0")
                sparsity = 0.0
            else:
                num_zeros = np.count_nonzero(flat_w == 0)
                sparsity = float(num_zeros) / float(total)
                print(f"[AnnotateMVAUSparsity]    total elems: {total}, zeros: {num_zeros}, sparsity: {sparsity:.6f}")

            # 如果节点上已有 sparsity 属性，先移除
            kept_attrs = []
            had_old_sparsity = False
            for attr in node.attribute:
                if attr.name == "sparsity":
                    had_old_sparsity = True
                else:
                    kept_attrs.append(attr)
            if had_old_sparsity:
                print("[AnnotateMVAUSparsity]    node already had 'sparsity' attribute -> replacing")

            # 清空再加回非 sparsity 的
            del node.attribute[:]
            for attr in kept_attrs:
                node.attribute.append(attr)

            # 加上新的 sparsity 属性
            node.attribute.append(helper.make_attribute("sparsity", sparsity))
            print("[AnnotateMVAUSparsity]    added attribute: sparsity =", sparsity)

            graph_modified = True

        print("[AnnotateMVAUSparsity] done, graph_modified =", graph_modified)
        return (model, False)
# ------------------------------------------------------------------------------#
# Name: SetFoldingSparsity
# Author: Changhong Li
# Date: Nov, 2025
# Description: Set folding attributes considering sparsity information
# # + derive mem_mode to avoid FIFO overflows
# ------------------------------------------------------------------------------#



import warnings
import numpy as np

# from finn.transformation.base import Transformation
# from finn.util.basic import (
#     is_hls_node,
#     is_rtl_node,
#     getCustomOp,
#     divisors,
#     common_divisors,
# )
from finn.transformation.fpgadataflow.set_folding import (
    GiveUniqueNodeNames,
    AnnotateCycles,
)
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance


class SetFoldingSparsity(Transformation):
    FIFO_LIMIT_BITS = 8191

    def __init__(self, target_cycles_per_frame=1000, mvau_wwidth_max=36, two_pass_relaxation=True):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation

    # 小工具：把 "INT2" / "UINT4" / "BIPOLAR" 这种转成 bitwidth
    def _weight_dtype_to_bw(self, wdt_str):
        if not isinstance(wdt_str, str):
            return None
        s = wdt_str.upper()
        # 最常见两类
        if s.startswith("INT"):
            # INT2 -> 2
            try:
                return int(s[3:])
            except Exception:
                return None
        if s.startswith("UINT"):
            try:
                return int(s[4:])
            except Exception:
                return None
        # 有些 FINN 节点可能写 BIPOLAR
        if s == "BIPOLAR":
            return 1
        return None

    def _set_mem_mode(self, node_inst):
        # 没有 mem_mode 就不用管
        try:
            attr_types = node_inst.get_nodeattr_types()
        except Exception:
            return
        if "mem_mode" not in attr_types:
            return

        # PE
        pe = 1
        if "PE" in attr_types:
            pe = node_inst.get_nodeattr("PE")
        elif "NumChannels" in attr_types:
            pe = node_inst.get_nodeattr("NumChannels")

        # SIMD
        simd = 1
        if "SIMD" in attr_types:
            simd = node_inst.get_nodeattr("SIMD")

        # BW：优先 weightDataType
        bw = 1
        if "weightDataType" in attr_types:
            wdt_str = node_inst.get_nodeattr("weightDataType")
            parsed = self._weight_dtype_to_bw(wdt_str)
            if parsed is not None:
                bw = parsed
            else:
                bw = 1
        else:
            # 没写 weightDataType 就退回到输入 dtype
            got_bw = False
            for port_id in [0, 1]:
                if got_bw:
                    break
                try:
                    dt = node_inst.get_input_datatype(port_id)
                    if dt is not None:
                        bw = dt.bitwidth()
                        got_bw = True
                except Exception:
                    pass
            if not got_bw:
                try:
                    dt = node_inst.get_output_datatype(0)
                    if dt is not None:
                        bw = dt.bitwidth()
                except Exception:
                    pass

        stream_bits = pe * simd * bw
        if stream_bits > self.FIFO_LIMIT_BITS:
            node_inst.set_nodeattr("mem_mode", "internal_embedded")
        else:
            node_inst.set_nodeattr("mem_mode", "internal_decoupled")

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.target_cycles_per_frame:
                break

    def apply(self, model):
        graph = model.graph
        pe_ops = [
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
            *ELEMENTWISE_BINARY_OPS,
        ]
        simd_ops = [
            "FMPadding_rtl",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_rtl",
            "StreamingSplit_hls",
            "StreamingConcat_hls",
        ]
        depthwise_op_exceptions = ["VVAU_hls", "VVAU_rtl", "Pool_hls"]

        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            op_type = node.op_type
            node_inst = getCustomOp(node)

            if op_type in ["MVAU_hls", "MVAU_rtl"]:
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")
                node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)
                for simd_val in divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.target_cycles_per_frame:
                        break
                    if (
                        node_inst.get_input_datatype(1).bitwidth() * node_inst.get_nodeattr("SIMD")
                        > self.mvau_wwidth_max
                    ):
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                self._set_mem_mode(node_inst)

            elif op_type in pe_ops:
                try:
                    max_pe = node_inst.get_nodeattr("NumChannels")
                except AttributeError:
                    max_pe = node_inst.get_normal_input_shape()[-1]
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                self._set_mem_mode(node_inst)

            elif op_type == "LabelSelect_hls":
                max_pe = node_inst.get_nodeattr("Labels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                self._set_mem_mode(node_inst)

            elif op_type in depthwise_op_exceptions:
                if op_type in ["VVAU_hls", "VVAU_rtl"]:
                    node_inst.set_nodeattr("SIMD", 1)
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                pe = node_inst.get_nodeattr("PE")
                cyc = node_inst.get_exp_cycles()
                if (
                    op_type in ["VVAU_hls", "VVAU_rtl"]
                    and pe == max_pe
                    and cyc > self.target_cycles_per_frame
                ):
                    max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")

                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type.startswith("ConvolutionInputGenerator"):
                    swu_node_inst = getCustomOp(swu_node)
                    swu_node_inst.set_nodeattr("SIMD", pe)
                    if swu_node.op_type == "ConvolutionInputGenerator_rtl":
                        if op_type.startswith("VVAU") and node_inst.get_nodeattr("SIMD") > 1:
                            swu_node_inst.set_nodeattr("parallel_window", 1)
                        else:
                            swu_node_inst.set_nodeattr("parallel_window", 0)
                else:
                    if op_type in ["VVAU_hls", "VVAU_rtl"]:
                        ksize = np.prod(node_inst.get_nodeattr("Kernel"))
                    elif op_type == "Pool_hls":
                        ksize = node_inst.get_nodeattr("KernelSize")
                    else:
                        raise Exception("Undefined edge case for %s" % op_type)
                    if ksize != 1:
                        raise Exception("Expected SWU on DW op input, found " + swu_node.op_type)
                self._set_mem_mode(node_inst)

            elif op_type in simd_ops:
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 0:
                        max_simd = node_inst.get_nodeattr("IFMChannels")
                        if op_type == "ConvolutionInputGenerator_rtl":
                            node_inst.set_nodeattr("parallel_window", 0)
                        self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                        simd = node_inst.get_nodeattr("SIMD")
                        cyc = node_inst.get_exp_cycles()
                        if (
                            op_type == "ConvolutionInputGenerator_rtl"
                            and simd == max_simd
                            and cyc > self.target_cycles_per_frame
                        ):
                            node_inst.set_nodeattr("parallel_window", 1)
                    self._set_mem_mode(node_inst)
                elif op_type in ["StreamingConcat_hls", "StreamingSplit_hls"]:
                    node_inst.set_nodeattr("SIMD", 1)
                    channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
                    for simd_val in common_divisors(channels_per_stream):
                        node_inst.set_nodeattr("SIMD", simd_val)
                        cyc = node_inst.get_exp_cycles()
                        if cyc < self.target_cycles_per_frame:
                            break
                    self._set_mem_mode(node_inst)
                else:
                    max_simd = node_inst.get_nodeattr("NumChannels")
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                    self._set_mem_mode(node_inst)
            else:
                warnings.warn("SetFolding doesn't know how to handle op_type " + op_type)
                self._set_mem_mode(node_inst)

        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        if self.two_pass_relaxation:
            perf_dict = model.analysis(dataflow_performance)
            if perf_dict["max_cycles"] > self.target_cycles_per_frame:
                warnings.warn(
                    "Node %s is bottleneck with %d cycles, running second pass"
                    % (perf_dict["max_cycles_node_name"], perf_dict["max_cycles"])
                )
                model = model.transform(
                    SetFoldingSparsity(
                        target_cycles_per_frame=perf_dict["max_cycles"],
                        mvau_wwidth_max=self.mvau_wwidth_max,
                        two_pass_relaxation=False,
                    )
                )

        return (model, False)


import numpy as np
from onnx import helper, AttributeProto


# ------------------------------------------------------------------------------#
# Name: SetMVAUSparseMode
# Author: Changhong Li
# Date: Nov, 2025
# Description: Set sparsity mode for MVAU nodes based on sparsity attribute
# ------------------------------------------------------------------------------#

class SetMVAUSparseModeHybrid(Transformation):
    """遍历所有 MVAU_hls 节点，新增/更新 sparse_mode 属性。
    规则：
      1) 若 MH == PE 且 MW == SIMD -> 'lut_sparse'
      2) 否则若 sparsity > 0.8 -> 'spmv_sparse'
      3) 否则 -> 'dense'
    注意：假定节点上已有 AnnotateMVAUSparsity 添加的 'sparsity' 属性。
    若缺失则按 0.0 处理。
    """
    def __init__(self, fpga_part=None):
        super().__init__()
        self.fpga_part = fpga_part

    def _get_attr(self, node, name, default=None):
        for attr in node.attribute:
            if attr.name != name:
                continue
            # 按类型安全读取
            if attr.type == AttributeProto.INT:
                return int(attr.i)
            if attr.type == AttributeProto.FLOAT:
                return float(attr.f)
            if attr.type == AttributeProto.STRING:
                s = attr.s
                try:
                    return s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                except Exception:
                    return str(s)
            if attr.type == AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == AttributeProto.FLOATS:
                return list(attr.floats)
            # 其它类型用不到，返回默认
            return default
        return default

    def _replace_attr(self, node, key, value):
        # 删除已有的同名属性
        kept = [a for a in node.attribute if a.name != key]
        del node.attribute[:]
        for a in kept:
            node.attribute.append(a)
        # 添加新属性
        node.attribute.append(helper.make_attribute(key, value))

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[SetMVAUSparseMode] start")
        print(f"[SetMVAUSparseMode] total nodes: {len(graph.node)}")

        for idx, node in enumerate(graph.node):
            if node.op_type != "MVAU_hls":
                continue

            print(f"\n[SetMVAUSparseMode] -> processing node #{idx}: name={node.name}, op_type={node.op_type}")

            # 读取所需属性
            # MH   = self._get_attr(node, "MH",   None)
            # MW   = self._get_attr(node, "MW",   None)
            # PE   = self._get_attr(node, "PE",   None)
            # SIMD = self._get_attr(node, "SIMD", None)
            # sparsity = self._get_attr(node, "sparsity", 0.0)

            # print(f"[SetMVAUSparseMode]    MH={MH}, MW={MW}, PE={PE}, SIMD={SIMD}, sparsity={sparsity}")

          
            # mode = "dense"
            # if (MH is not None and PE is not None and MW is not None and SIMD is not None
            #     and MH == PE and MW == SIMD):
            #     mode = "lut_sparse"

            #     mem_mode = 'internal_embedded'

            #     self._replace_attr(node, "mem_mode", mem_mode)

            #     reason = "MH==PE && MW==SIMD"
            # elif sparsity is not None and float(sparsity) > 0.5:
            #     mode = "spmv_sparse"
            #     reason = "sparsity>0.5"
            # else:
            #     reason = "fallback dense"

            # print(f"[SetMVAUSparseMode]    set sparse_mode='{mode}' ({reason})")

            MH   = self._get_attr(node, "MH",   None)
            MW   = self._get_attr(node, "MW",   None)
            PE   = self._get_attr(node, "PE",   None)
            SIMD = self._get_attr(node, "SIMD", None)
            sparsity = self._get_attr(node, "sparsity", 0.0)
            tile_sparsity = self._get_attr(node, "tile_sparsity", 0.0)

            print(f"[SetMVAUSparseMode] MH={MH}, MW={MW}, PE={PE}, SIMD={SIMD}, "
                f"sparsity={sparsity}, tile_sparsity={tile_sparsity}")

            mode = "dense"
            mem_mode = None

            unfold_lutsp_ok = False
            # resource estimation before change            
            node_inst = getCustomOp(node)
            res_dict = {}
            res_dict[node.name] = node_inst.node_res_estimation(self.fpga_part)
            res = res_dict[node.name]
            # get lut value
            lut_val = int(res.get("LUT", 0))
            print(f"[SetMVAUSparseMode]    estimated LUTs: {lut_val}")
            self._replace_attr(node, "PE", MH)
            self._replace_attr(node, "SIMD", MW)
            node_inst = getCustomOp(node)
            res_dict = {}
            res_dict[node.name] = node_inst.node_res_estimation(self.fpga_part)
            res = res_dict[node.name]
            # get lut value
            lut_val_new = int(res.get("LUT", 0))
            print(f"[SetMVAUSparseMode]    estimated UNFOLD LUTs: {lut_val_new}")
            # check if lut sp is ok?
            sparsed_lut = lut_val_new * (1 - sparsity) * 1.95
            print(f"[SetMVAUSparseMode]    estimated sparse unfold LUTs: {sparsed_lut}")
            if (sparsed_lut < lut_val) and (MW * MH < 38500):
                unfold_lutsp_ok = True
                print(f"[SetMVAUSparseMode]    unfold lut sparse mode is OK")
            else:
                unfold_lutsp_ok = False
                print(f"[SetMVAUSparseMode]    unfold lut sparse mode is NOT OK")
            # revert back
            self._replace_attr(node, "PE", PE)
            self._replace_attr(node, "SIMD", SIMD)








            # 1) 全局稀疏度 < 0.6，直接 dense
            if sparsity is not None and float(sparsity) < 0.6:
                reason = "sparsity<0.6 -> dense"

            # 2) 判断是否全展开：MH==PE && MW==SIMD，全展开直接 lut_sparse
            elif (MH is not None and PE is not None and
                MW is not None and SIMD is not None and
                MH == PE and MW == SIMD):
                mode = "lut_sparse"
                mem_mode = "internal_embedded"
                self._replace_attr(node, "mem_mode", mem_mode)
                reason = "fully unrolled: MH==PE && MW==SIMD"

            # 3) 按 tile_sparsity 决定：>0.7 -> spmv_sparse
            elif tile_sparsity is not None and float(tile_sparsity) > 0.7:
                mode = "spmv_sparse"
                reason = "tile_sparsity>0.7"
            # 4) CHECK IF LUT SP OK?
            elif unfold_lutsp_ok:
                self._replace_attr(node, "PE", MH)
                self._replace_attr(node, "SIMD", MW)
                mode = "lut_sparse"
                mem_mode = "internal_embedded"
                self._replace_attr(node, "mem_mode", mem_mode)
                reason = "fully unrolled: unfold lut sparse mode is OK"
            # 5) 其他情况弹回 dense
            else:
                reason = "fallback dense"


            print(f"[SetMVAUSparseMode] mode={mode}, reason={reason}")


            # 更新/新增 sparse_mode
            self._replace_attr(node, "sparse_mode", mode)
            graph_modified = True

        print("[SetMVAUSparseMode] done, graph_modified =", graph_modified)
        return (model, False)

class SetMVAUSparseMode(Transformation):
    """遍历所有 MVAU_hls 节点，新增/更新 sparse_mode 属性。
    规则：
      1) 若 MH == PE 且 MW == SIMD -> 'lut_sparse'
      2) 否则若 sparsity > 0.8 -> 'spmv_sparse'
      3) 否则 -> 'dense'
    注意：假定节点上已有 AnnotateMVAUSparsity 添加的 'sparsity' 属性。
    若缺失则按 0.0 处理。
    """

    def _get_attr(self, node, name, default=None):
        for attr in node.attribute:
            if attr.name != name:
                continue
            # 按类型安全读取
            if attr.type == AttributeProto.INT:
                return int(attr.i)
            if attr.type == AttributeProto.FLOAT:
                return float(attr.f)
            if attr.type == AttributeProto.STRING:
                s = attr.s
                try:
                    return s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                except Exception:
                    return str(s)
            if attr.type == AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == AttributeProto.FLOATS:
                return list(attr.floats)
            # 其它类型用不到，返回默认
            return default
        return default

    def _replace_attr(self, node, key, value):
        # 删除已有的同名属性
        kept = [a for a in node.attribute if a.name != key]
        del node.attribute[:]
        for a in kept:
            node.attribute.append(a)
        # 添加新属性
        node.attribute.append(helper.make_attribute(key, value))

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[SetMVAUSparseMode] start")
        print(f"[SetMVAUSparseMode] total nodes: {len(graph.node)}")

        for idx, node in enumerate(graph.node):
            if node.op_type != "MVAU_hls":
                continue

            print(f"\n[SetMVAUSparseMode] -> processing node #{idx}: name={node.name}, op_type={node.op_type}")

            # 读取所需属性
            MH   = self._get_attr(node, "MH",   None)
            MW   = self._get_attr(node, "MW",   None)
            PE   = self._get_attr(node, "PE",   None)
            SIMD = self._get_attr(node, "SIMD", None)
            sparsity = self._get_attr(node, "sparsity", 0.0)

            print(f"[SetMVAUSparseMode]    MH={MH}, MW={MW}, PE={PE}, SIMD={SIMD}, sparsity={sparsity}")

            # 判定模式
            mode = "dense"
            if (MH is not None and PE is not None and MW is not None and SIMD is not None
                and MH == PE and MW == SIMD):
                mode = "lut_sparse"
                reason = "MH==PE && MW==SIMD"
            elif sparsity is not None and float(sparsity) > 0.8:
                mode = "spmv_sparse"
                reason = "sparsity>0.8"
            else:
                reason = "fallback dense"

            print(f"[SetMVAUSparseMode]    set sparse_mode='{mode}' ({reason})")

            # 更新/新增 sparse_mode
            self._replace_attr(node, "sparse_mode", mode)
            graph_modified = True

        print("[SetMVAUSparseMode] done, graph_modified =", graph_modified)
        return (model, False)
    
class SetMVAUSparseMode_spmvonly(Transformation):
    """遍历所有 MVAU_hls 节点，新增/更新 sparse_mode 属性。
    规则：
      1) 若 MH == PE 且 MW == SIMD -> 'lut_sparse'
      2) 否则若 sparsity > 0.8 -> 'spmv_sparse'
      3) 否则 -> 'dense'
    注意：假定节点上已有 AnnotateMVAUSparsity 添加的 'sparsity' 属性。
    若缺失则按 0.0 处理。
    """

    def _get_attr(self, node, name, default=None):
        for attr in node.attribute:
            if attr.name != name:
                continue
            # 按类型安全读取
            if attr.type == AttributeProto.INT:
                return int(attr.i)
            if attr.type == AttributeProto.FLOAT:
                return float(attr.f)
            if attr.type == AttributeProto.STRING:
                s = attr.s
                try:
                    return s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                except Exception:
                    return str(s)
            if attr.type == AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == AttributeProto.FLOATS:
                return list(attr.floats)
            # 其它类型用不到，返回默认
            return default
        return default

    def _replace_attr(self, node, key, value):
        # 删除已有的同名属性
        kept = [a for a in node.attribute if a.name != key]
        del node.attribute[:]
        for a in kept:
            node.attribute.append(a)
        # 添加新属性
        node.attribute.append(helper.make_attribute(key, value))

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[SetMVAUSparseMode] start")
        print(f"[SetMVAUSparseMode] total nodes: {len(graph.node)}")

        for idx, node in enumerate(graph.node):
            if node.op_type != "MVAU_hls":
                continue

            print(f"\n[SetMVAUSparseMode] -> processing node #{idx}: name={node.name}, op_type={node.op_type}")

            # 读取所需属性
            MH   = self._get_attr(node, "MH",   None)
            MW   = self._get_attr(node, "MW",   None)
            PE   = self._get_attr(node, "PE",   None)
            SIMD = self._get_attr(node, "SIMD", None)
            sparsity = self._get_attr(node, "sparsity", 0.0)

            print(f"[SetMVAUSparseMode]    MH={MH}, MW={MW}, PE={PE}, SIMD={SIMD}, sparsity={sparsity}")

            # # 判定模式
            # mode = "dense"
            # if (MH is not None and PE is not None and MW is not None and SIMD is not None
            #     and MH == PE and MW == SIMD):
            #     mode = "lut_sparse"
            #     reason = "MH==PE && MW==SIMD"
            # elif sparsity is not None and float(sparsity) > 0.8:
            #     mode = "spmv_sparse"
            #     reason = "sparsity>0.8"
            # else:
            #     reason = "fallback dense"
            mode = "spmv_sparse"
            reason = "force spmv_sparse"

            print(f"[SetMVAUSparseMode]    set sparse_mode='{mode}' ({reason})")

            # 更新/新增 sparse_mode
            self._replace_attr(node, "sparse_mode", mode)
            graph_modified = True

        print("[SetMVAUSparseMode] done, graph_modified =", graph_modified)
        return (model, False)
    
class SetMVAUSparseMode_lutsponly(Transformation):
    """遍历所有 MVAU_hls 节点，新增/更新 sparse_mode 属性。
    规则：
      1) 若 MH == PE 且 MW == SIMD -> 'lut_sparse'
      2) 否则若 sparsity > 0.8 -> 'spmv_sparse'
      3) 否则 -> 'dense'
    注意：假定节点上已有 AnnotateMVAUSparsity 添加的 'sparsity' 属性。
    若缺失则按 0.0 处理。
    """

    def _get_attr(self, node, name, default=None):
        for attr in node.attribute:
            if attr.name != name:
                continue
            # 按类型安全读取
            if attr.type == AttributeProto.INT:
                return int(attr.i)
            if attr.type == AttributeProto.FLOAT:
                return float(attr.f)
            if attr.type == AttributeProto.STRING:
                s = attr.s
                try:
                    return s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                except Exception:
                    return str(s)
            if attr.type == AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == AttributeProto.FLOATS:
                return list(attr.floats)
            # 其它类型用不到，返回默认
            return default
        return default

    def _replace_attr(self, node, key, value):
        # 删除已有的同名属性
        kept = [a for a in node.attribute if a.name != key]
        del node.attribute[:]
        for a in kept:
            node.attribute.append(a)
        # 添加新属性
        node.attribute.append(helper.make_attribute(key, value))

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[SetMVAUSparseMode] start")
        print(f"[SetMVAUSparseMode] total nodes: {len(graph.node)}")

        for idx, node in enumerate(graph.node):
            if node.op_type != "MVAU_hls":
                continue

            print(f"\n[SetMVAUSparseMode] -> processing node #{idx}: name={node.name}, op_type={node.op_type}")

            # 读取所需属性
            MH   = self._get_attr(node, "MH",   None)
            MW   = self._get_attr(node, "MW",   None)
            PE   = self._get_attr(node, "PE",   None)
            SIMD = self._get_attr(node, "SIMD", None)
            sparsity = self._get_attr(node, "sparsity", 0.0)

            print(f"[SetMVAUSparseMode]    MH={MH}, MW={MW}, PE={PE}, SIMD={SIMD}, sparsity={sparsity}")

            # # 判定模式
            # mode = "dense"
            # if (MH is not None and PE is not None and MW is not None and SIMD is not None
            #     and MH == PE and MW == SIMD):
            #     mode = "lut_sparse"
            #     reason = "MH==PE && MW==SIMD"
            # elif sparsity is not None and float(sparsity) > 0.8:
            #     mode = "spmv_sparse"
            #     reason = "sparsity>0.8"
            # else:
            #     reason = "fallback dense"
            mode = "lut_sparse"
            mem_mode = 'internal_embedded'
            reason = "force lut_sparse"

            print(f"[SetMVAUSparseMode]    set sparse_mode='{mode}' ({reason})")

            # 更新/新增 sparse_mode
            self._replace_attr(node, "sparse_mode", mode)
            self._replace_attr(node, "mem_mode", mem_mode)
            graph_modified = True

        print("[SetMVAUSparseMode] done, graph_modified =", graph_modified)
        return (model, False)

class AnnotateMVAUTileSparsity(Transformation):
    """在图中找到所有 MVAU_hls 节点，对其权重(输入2)按(PE, SIMD) tile进行稀疏度分析并写回到节点属性中"""

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[AnnotateMVAUTileSparsity] start")
        print(f"[AnnotateMVAUTileSparsity] total nodes: {len(graph.node)}")

        for idx, node in enumerate(graph.node):
            # 只处理 MVAU_hls
            if node.op_type != "MVAU_hls":
                continue

            print(f"\n[AnnotateMVAUTileSparsity] -> processing node #{idx}: name={node.name}, op_type={node.op_type}")

            # MVAU 通常至少有 3 个输入: data, ..., weights
            if len(node.input) < 3:
                print(f"[AnnotateMVAUTileSparsity]    skip: node has only {len(node.input)} inputs, no weight at index 2")
                continue

            weight_name = node.input[1]
            print(f"[AnnotateMVAUTileSparsity]    weight input name: {weight_name}")

            # 读取 PE / SIMD
            pe = None
            simd = None
            for attr in node.attribute:
                if attr.name == "PE":
                    pe = int(attr.i) if hasattr(attr, "i") else None
                elif attr.name == "SIMD":
                    simd = int(attr.i) if hasattr(attr, "i") else None

            if pe is None or simd is None or pe <= 0 or simd <= 0:
                print(f"[AnnotateMVAUTileSparsity]    skip: invalid PE/SIMD (PE={pe}, SIMD={simd})")
                continue

            # 获取权重
            weight_arr = model.get_initializer(weight_name)
            if weight_arr is None:
                print(f"[AnnotateMVAUTileSparsity]    skip: weight '{weight_name}' is not an initializer (maybe runtime)")
                continue

            w_np = weight_arr if isinstance(weight_arr, np.ndarray) else np.array(weight_arr)
            print(f"[AnnotateMVAUTileSparsity]    weight shape: {w_np.shape}")

            # ---- 将权重展开为二维矩阵 [out_dim, in_dim] ----
            if w_np.ndim == 0:
                w2d = w_np.reshape(1, 1)
            elif w_np.ndim == 1:
                w2d = w_np.reshape(-1, 1)
            elif w_np.ndim == 2:
                w2d = w_np
            else:
                in_dim = w_np.shape[-1]
                out_dim = int(np.prod(w_np.shape[:-1]))
                w2d = w_np.reshape(out_dim, in_dim)

            out_dim, in_dim = w2d.shape
            print(f"[AnnotateMVAUTileSparsity]    2D view shape: {w2d.shape}, PE={pe}, SIMD={simd}")

            total_tiles = 0
            zero_tiles = 0

            # 遍历 tile
            for r in range(0, out_dim, pe):
                for c in range(0, in_dim, simd):
                    tile = w2d[r:min(r + pe, out_dim), c:min(c + simd, in_dim)]
                    total_tiles += 1
                    if np.count_nonzero(tile) == 0:
                        zero_tiles += 1

            if total_tiles == 0:
                print("[AnnotateMVAUTileSparsity]    warning: no tiles found, set tile_sparsity=0.0")
                tile_sparsity = 0.0
            else:
                tile_sparsity = float(zero_tiles) / float(total_tiles)

            print(f"[AnnotateMVAUTileSparsity]    tiles: total={total_tiles}, zero_tiles={zero_tiles}, tile_sparsity={tile_sparsity:.6f}")

            # 清理旧的 tile_sparsity 属性
            kept_attrs = []
            had_old = False
            for attr in node.attribute:
                if attr.name == "tile_sparsity":
                    had_old = True
                else:
                    kept_attrs.append(attr)
            if had_old:
                print("[AnnotateMVAUTileSparsity]    node already had 'tile_sparsity' attribute -> replacing")

            del node.attribute[:]
            for attr in kept_attrs:
                node.attribute.append(attr)

            # 添加新的属性
            node.attribute.append(helper.make_attribute("tile_sparsity", tile_sparsity))
            print("[AnnotateMVAUTileSparsity]    added attribute: tile_sparsity =", tile_sparsity)

            graph_modified = True

        print("[AnnotateMVAUTileSparsity] done, graph_modified =", graph_modified)
        return (model, False)

class GlobalPruneMBv1Weights90(Transformation):
    def __init__(self, prune_ratio=0.98, min_nonzero_per_layer=32):
        super().__init__()
        self.prune_ratio = prune_ratio
        self.min_nonzero_per_layer = min_nonzero_per_layer

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[GlobalPruneMVAUWeights90] start")
        print(f"[GlobalPruneMVAUWeights90] total nodes: {len(graph.node)}")
        print(f"[GlobalPruneMVAUWeights90] prune_ratio={self.prune_ratio}, "
              f"min_nonzero_per_layer={self.min_nonzero_per_layer}")

        # ---------- 第 1 遍：收集所有 MVAU_hls 的权重，做全局阈值统计 ----------
        weight_infos = []   # 每个元素: dict(node_idx, node_name, weight_name, w_np)
        all_abs_flat_list = []

        for idx, node in enumerate(graph.node):
            if node.op_type != "MVAU_hls":
                continue
            # node_inst = getCustomOp(node)
            # if node_inst.get_nodeattr("MW")
            print(f"\n[GlobalPruneMVAUWeights90] -> found MVAU_hls node #{idx}: "
                  f"name={node.name}, op_type={node.op_type}")

            if len(node.input) < 2:
                print(f"[GlobalPruneMVAUWeights90]    skip: node has only "
                      f"{len(node.input)} inputs, no weight at index 1")
                continue

            weight_name = node.input[1]
            print(f"[GlobalPruneMVAUWeights90]    weight input name: {weight_name}")

            w_arr = model.get_initializer(weight_name)
            if w_arr is None:
                print(f"[GlobalPruneMVAUWeights90]    skip: weight '{weight_name}' "
                      f"is not an initializer (maybe runtime)")
                continue

            w_np = w_arr
            if not isinstance(w_np, np.ndarray):
                w_np = np.array(w_arr)

            print(f"[GlobalPruneMVAUWeights90]    weight shape: {w_np.shape}, "
                  f"dtype={w_np.dtype}")

            flat_abs = np.abs(w_np).flatten().astype(np.float64)
            if flat_abs.size == 0:
                print("[GlobalPruneMVAUWeights90]    warning: weight tensor has 0 elements")
                continue

            all_abs_flat_list.append(flat_abs)
            weight_infos.append({
                "node_idx": idx,
                "node_name": node.name,
                "weight_name": weight_name,
                "w_np": w_np,
            })

        if len(weight_infos) == 0:
            print("[GlobalPruneMVAUWeights90] no MVAU_hls weights found, nothing to prune.")
            return (model, False)

        all_abs_flat = np.concatenate(all_abs_flat_list, axis=0)
        total_elems = all_abs_flat.size
        print(f"\n[GlobalPruneMVAUWeights90] collected total elements: {total_elems}")

        if total_elems == 0:
            print("[GlobalPruneMVAUWeights90] total elements is 0, nothing to prune.")
            return (model, False)

        keep_ratio = 1.0 - self.prune_ratio
        if keep_ratio <= 0.0:
            print("[GlobalPruneMVAUWeights90] keep_ratio <= 0, force keep_ratio=1e-6")
            keep_ratio = 1e-6

        # 计算“保留 top-k”对应的全局阈值：
        #   目标是保留 top (keep_ratio) 的权重，等价于用百分位数 100 * (1 - keep_ratio)
        #   对于 90% 剪枝（keep_ratio=0.1），就是 90 百分位
        percentile = 100.0 * (1.0 - keep_ratio)
        global_threshold = np.percentile(all_abs_flat, percentile)

        print(f"[GlobalPruneMVAUWeights90] global threshold (percentile={percentile:.2f}): "
              f"{global_threshold:.6e}")

        # ---------- 第 2 遍：按全局阈值剪枝，并保证每层至少保留 min_nonzero_per_layer ----------
        global_old_nonzero = 0
        global_new_nonzero = 0

        for info in weight_infos:
            idx = info["node_idx"]
            node_name = info["node_name"]
            weight_name = info["weight_name"]
            w_np = info["w_np"]

            print(f"\n[GlobalPruneMVAUWeights90] -> pruning node #{idx}, name={node_name}, "
                  f"weight={weight_name}")

            flat = w_np.flatten()
            abs_flat = np.abs(flat)

            total = flat.size
            old_nonzero = np.count_nonzero(flat)
            global_old_nonzero += old_nonzero

            if total == 0:
                print("[GlobalPruneMVAUWeights90]    skip: weight has 0 elements")
                continue

            # 初步：按全局阈值，保留 abs > threshold 的权重
            keep_mask = abs_flat > global_threshold
            num_keep_initial = int(np.count_nonzero(keep_mask))

            print(f"[GlobalPruneMVAUWeights90]    total={total}, "
                  f"old_nonzero={old_nonzero}, "
                  f"keep_by_threshold={num_keep_initial}")

            # 约束 1：如果该层元素总数小于 min_nonzero_per_layer，
            #         则没法“保留至少 64 个非零”——直接全部保留（不做剪枝）
            if total < self.min_nonzero_per_layer:
                print(f"[GlobalPruneMVAUWeights90]    total({total}) < "
                      f"min_nonzero_per_layer({self.min_nonzero_per_layer}), "
                      f"skip pruning for this weight (keep all).")
                keep_mask = np.ones_like(flat, dtype=bool)
            else:
                # 约束 2：确保每层至少保留 min_nonzero_per_layer 个非零权重
                if num_keep_initial < self.min_nonzero_per_layer:
                    print(f"[GlobalPruneMVAUWeights90]    keep_by_threshold({num_keep_initial}) "
                          f"< min_nonzero_per_layer({self.min_nonzero_per_layer}), "
                          f"force keeping top-{self.min_nonzero_per_layer} by magnitude.")

                    # 找到该层中按绝对值排序的 top-k 下标
                    k = self.min_nonzero_per_layer
                    # 使用 argpartition 实现 O(n) 级别的 top-k
                    # 排序按 -abs_flat，即绝对值从大到小
                    topk_idx = np.argpartition(-abs_flat, k - 1)[:k]
                    keep_mask[:] = False
                    keep_mask[topk_idx] = True

            # 应用剪枝：keep_mask 为 True 的保持原值，False 的置 0
            pruned_flat = flat.copy()
            pruned_flat[~keep_mask] = 0

            new_nonzero = int(np.count_nonzero(pruned_flat))
            global_new_nonzero += new_nonzero

            new_sparsity = 1.0 - float(new_nonzero) / float(total)
            print(f"[GlobalPruneMVAUWeights90]    new_nonzero={new_nonzero}, "
                  f"new_sparsity={new_sparsity:.6f}")

            # reshape 回原来的形状并保持 dtype
            pruned_w = pruned_flat.reshape(w_np.shape).astype(w_np.dtype)

            # 写回到模型的 initializer
            model.set_initializer(weight_name, pruned_w)
            graph_modified = True

        # ---------- 全局统计信息 ----------
        if global_old_nonzero == 0:
            print("\n[GlobalPruneMVAUWeights90] warning: global_old_nonzero == 0 (all weights already zero?)")
        else:
            global_old_sparsity = 1.0 - float(global_old_nonzero) / float(total_elems)
            global_new_sparsity = 1.0 - float(global_new_nonzero) / float(total_elems)
            print("\n[GlobalPruneMVAUWeights90] global stats:")
            print(f"    total_elems       = {total_elems}")
            print(f"    old_nonzero       = {global_old_nonzero}")
            print(f"    old_sparsity      = {global_old_sparsity:.6f}")
            print(f"    new_nonzero       = {global_new_nonzero}")
            print(f"    new_sparsity      = {global_new_sparsity:.6f}")
            print(f"    target_prune_ratio= {self.prune_ratio:.6f} "
                  "(note: per-layer min_nonzero may slightly deviate from exact ratio)")

        print("\n[GlobalPruneMVAUWeights90] done, graph_modified =", graph_modified)
        return (model, False)



class GlobalPruneMVAUWeights90(Transformation):
    """
    在图中找到所有 MVAU_hls 节点，对其权重做**全局 90% 剪枝**（按绝对值大小全局排序），
    并确保每一层保留的非零权重不少于 min_nonzero_per_layer 个，
    然后将剪枝后的权重写回到 ONNX 图中。
    """

    def __init__(self, prune_ratio=0.9, min_nonzero_per_layer=64):
        super().__init__()
        self.prune_ratio = prune_ratio
        self.min_nonzero_per_layer = min_nonzero_per_layer

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        print("[GlobalPruneMVAUWeights90] start")
        print(f"[GlobalPruneMVAUWeights90] total nodes: {len(graph.node)}")
        print(f"[GlobalPruneMVAUWeights90] prune_ratio={self.prune_ratio}, "
              f"min_nonzero_per_layer={self.min_nonzero_per_layer}")

        # ---------- 第 1 遍：收集所有 MVAU_hls 的权重，做全局阈值统计 ----------
        weight_infos = []   # 每个元素: dict(node_idx, node_name, weight_name, w_np)
        all_abs_flat_list = []

        for idx, node in enumerate(graph.node):
            if node.op_type != "MVAU_hls":
                continue

            print(f"\n[GlobalPruneMVAUWeights90] -> found MVAU_hls node #{idx}: "
                  f"name={node.name}, op_type={node.op_type}")

            if len(node.input) < 2:
                print(f"[GlobalPruneMVAUWeights90]    skip: node has only "
                      f"{len(node.input)} inputs, no weight at index 1")
                continue

            weight_name = node.input[1]
            print(f"[GlobalPruneMVAUWeights90]    weight input name: {weight_name}")

            w_arr = model.get_initializer(weight_name)
            if w_arr is None:
                print(f"[GlobalPruneMVAUWeights90]    skip: weight '{weight_name}' "
                      f"is not an initializer (maybe runtime)")
                continue

            w_np = w_arr
            if not isinstance(w_np, np.ndarray):
                w_np = np.array(w_arr)

            print(f"[GlobalPruneMVAUWeights90]    weight shape: {w_np.shape}, "
                  f"dtype={w_np.dtype}")

            flat_abs = np.abs(w_np).flatten().astype(np.float64)
            if flat_abs.size == 0:
                print("[GlobalPruneMVAUWeights90]    warning: weight tensor has 0 elements")
                continue

            all_abs_flat_list.append(flat_abs)
            weight_infos.append({
                "node_idx": idx,
                "node_name": node.name,
                "weight_name": weight_name,
                "w_np": w_np,
            })

        if len(weight_infos) == 0:
            print("[GlobalPruneMVAUWeights90] no MVAU_hls weights found, nothing to prune.")
            return (model, False)

        all_abs_flat = np.concatenate(all_abs_flat_list, axis=0)
        total_elems = all_abs_flat.size
        print(f"\n[GlobalPruneMVAUWeights90] collected total elements: {total_elems}")

        if total_elems == 0:
            print("[GlobalPruneMVAUWeights90] total elements is 0, nothing to prune.")
            return (model, False)

        keep_ratio = 1.0 - self.prune_ratio
        if keep_ratio <= 0.0:
            print("[GlobalPruneMVAUWeights90] keep_ratio <= 0, force keep_ratio=1e-6")
            keep_ratio = 1e-6

        # 计算“保留 top-k”对应的全局阈值：
        #   目标是保留 top (keep_ratio) 的权重，等价于用百分位数 100 * (1 - keep_ratio)
        #   对于 90% 剪枝（keep_ratio=0.1），就是 90 百分位
        percentile = 100.0 * (1.0 - keep_ratio)
        global_threshold = np.percentile(all_abs_flat, percentile)

        print(f"[GlobalPruneMVAUWeights90] global threshold (percentile={percentile:.2f}): "
              f"{global_threshold:.6e}")

        # ---------- 第 2 遍：按全局阈值剪枝，并保证每层至少保留 min_nonzero_per_layer ----------
        global_old_nonzero = 0
        global_new_nonzero = 0

        for info in weight_infos:
            idx = info["node_idx"]
            node_name = info["node_name"]
            weight_name = info["weight_name"]
            w_np = info["w_np"]

            print(f"\n[GlobalPruneMVAUWeights90] -> pruning node #{idx}, name={node_name}, "
                  f"weight={weight_name}")

            flat = w_np.flatten()
            abs_flat = np.abs(flat)

            total = flat.size
            old_nonzero = np.count_nonzero(flat)
            global_old_nonzero += old_nonzero

            if total == 0:
                print("[GlobalPruneMVAUWeights90]    skip: weight has 0 elements")
                continue

            # 初步：按全局阈值，保留 abs > threshold 的权重
            keep_mask = abs_flat > global_threshold
            num_keep_initial = int(np.count_nonzero(keep_mask))

            print(f"[GlobalPruneMVAUWeights90]    total={total}, "
                  f"old_nonzero={old_nonzero}, "
                  f"keep_by_threshold={num_keep_initial}")

            # 约束 1：如果该层元素总数小于 min_nonzero_per_layer，
            #         则没法“保留至少 64 个非零”——直接全部保留（不做剪枝）
            if total < self.min_nonzero_per_layer:
                print(f"[GlobalPruneMVAUWeights90]    total({total}) < "
                      f"min_nonzero_per_layer({self.min_nonzero_per_layer}), "
                      f"skip pruning for this weight (keep all).")
                keep_mask = np.ones_like(flat, dtype=bool)
            else:
                # 约束 2：确保每层至少保留 min_nonzero_per_layer 个非零权重
                if num_keep_initial < self.min_nonzero_per_layer:
                    print(f"[GlobalPruneMVAUWeights90]    keep_by_threshold({num_keep_initial}) "
                          f"< min_nonzero_per_layer({self.min_nonzero_per_layer}), "
                          f"force keeping top-{self.min_nonzero_per_layer} by magnitude.")

                    # 找到该层中按绝对值排序的 top-k 下标
                    k = self.min_nonzero_per_layer
                    # 使用 argpartition 实现 O(n) 级别的 top-k
                    # 排序按 -abs_flat，即绝对值从大到小
                    topk_idx = np.argpartition(-abs_flat, k - 1)[:k]
                    keep_mask[:] = False
                    keep_mask[topk_idx] = True

            # 应用剪枝：keep_mask 为 True 的保持原值，False 的置 0
            pruned_flat = flat.copy()
            pruned_flat[~keep_mask] = 0

            new_nonzero = int(np.count_nonzero(pruned_flat))
            global_new_nonzero += new_nonzero

            new_sparsity = 1.0 - float(new_nonzero) / float(total)
            print(f"[GlobalPruneMVAUWeights90]    new_nonzero={new_nonzero}, "
                  f"new_sparsity={new_sparsity:.6f}")

            # reshape 回原来的形状并保持 dtype
            pruned_w = pruned_flat.reshape(w_np.shape).astype(w_np.dtype)

            # 写回到模型的 initializer
            model.set_initializer(weight_name, pruned_w)
            graph_modified = True

        # ---------- 全局统计信息 ----------
        if global_old_nonzero == 0:
            print("\n[GlobalPruneMVAUWeights90] warning: global_old_nonzero == 0 (all weights already zero?)")
        else:
            global_old_sparsity = 1.0 - float(global_old_nonzero) / float(total_elems)
            global_new_sparsity = 1.0 - float(global_new_nonzero) / float(total_elems)
            print("\n[GlobalPruneMVAUWeights90] global stats:")
            print(f"    total_elems       = {total_elems}")
            print(f"    old_nonzero       = {global_old_nonzero}")
            print(f"    old_sparsity      = {global_old_sparsity:.6f}")
            print(f"    new_nonzero       = {global_new_nonzero}")
            print(f"    new_sparsity      = {global_new_sparsity:.6f}")
            print(f"    target_prune_ratio= {self.prune_ratio:.6f} "
                  "(note: per-layer min_nonzero may slightly deviate from exact ratio)")

        print("\n[GlobalPruneMVAUWeights90] done, graph_modified =", graph_modified)
        return (model, False)
