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
import math
import numpy as np
import onnx.numpy_helper as np_helper
import os
import qonnx.custom_op.general.xnorpopcount as xp
import textwrap
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string

# ONNX i/o tensor shape assumptions for MatrixVectorActivation:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)


class MVAU(HWCustomOp):
    """Abstraction layer for HW implementation of MatrixVectorActivation layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "MW": ("i", True, 0),
            "MH": ("i", True, 0),
            "resType": ("s", False, "auto", {"auto", "lut", "dsp"}),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # FINN DataType for accumulator -- auto-computed and updated
            "accDataType": ("s", False, "INT32"),
            # use xnor-popcount for binary weights/inputs, thus treating them
            # as bipolar
            "binaryXnorMode": ("i", False, 0, {0, 1}),
            # no-activation mode (produce accumulators)
            "noActivation": ("i", False, 0, {0, 1}),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # memory mode for the FC weights
            # internal_embedded -- embedded weights, long compile/synth times
            # internal_decoupled -- default, streaming weights with streamer packaged inside IP
            # external -- streaming weights with external streamer
            "mem_mode": (
                "s",
                False,
                "internal_decoupled",
                {"internal_embedded", "internal_decoupled", "external"},
            ),
            # FPGA resource type for memories in internal_decoupled mode
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use UltraRAM (URAM), must have runtime_writeable_weights=1
            # see also https://www.xilinx.com/support/answers/38070.html
            "ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            # FPGA resource type for threshold memories (if noActivation is False)
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            "ram_style_thresholds": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed"},
            ),
            # (mem_mode = internal_decoupled only) whether weights will be
            # writeable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "pumpedMemory": ("i", False, 0, {0, 1}),
            # dynamic input
            "dynamic_input": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def execute_node(self, context, graph):
        node = self.onnx_node
        in_act = context[node.input[0]]
        # ensure that shape is compatible
        in_act = in_act.reshape(self.get_normal_input_shape())

        if self.get_nodeattr("dynamic_input"):
            mvau_w = context[node.input[1]]
        else:
            mvau_w_init = [x for x in graph.initializer if x.name == node.input[1]][0]
            mvau_w = np_helper.to_array(mvau_w_init)

        # Matrix multiplication
        if self.get_nodeattr("binaryXnorMode"):
            # Note: activation/weights are expected to be binary
            # (by design coming from the transformation inferring this operation mode)
            result = xp.xnorpopcountmatmul(in_act, mvau_w)
        elif (
            self.get_nodeattr("inputDataType") == "BIPOLAR"
            and self.get_nodeattr("weightDataType") == "BIPOLAR"
        ):
            # Convert to binary and use xnorpopcountmatmul function
            result = xp.xnorpopcountmatmul((in_act + 1) / 2, (mvau_w + 1) / 2)
        else:
            # Regular matrix multiplication
            result = np.matmul(in_act, mvau_w)
        if self.get_nodeattr("noActivation") == 0:
            mvau_thr_init = [x for x in graph.initializer if x.name == node.input[2]][0]
            mvau_thr = np_helper.to_array(mvau_thr_init)
            odt_is_bipolar = self.get_nodeattr("outputDataType") == "BIPOLAR"
            out_scale = 2 if odt_is_bipolar else 1
            out_bias = -1 if odt_is_bipolar else self.get_nodeattr("ActVal")
            if result.ndim == 4:
                # NHWC to NCHW for multithreshold node
                result = result.transpose((0, 3, 1, 2))
            result = multithreshold(result, mvau_thr, out_scale, out_bias)
            if result.ndim == 4:
                # NCHW to NHWC
                result = result.transpose((0, 2, 3, 1))
        oshape = context[node.output[0]].shape
        context[node.output[0]] = result.reshape(oshape)

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        # TODO collect automatically from get_nodeattr_types
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("resType")
            self.get_nodeattr("MW")
            self.get_nodeattr("MH")
            self.get_nodeattr("SIMD")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("weightDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required MatrixVectorActivation attributes do not exist.""")

        # verify the number of inputs depending on noActivation value
        # check noActivation value to determine the number of inputs
        no_act = self.get_nodeattr("noActivation")

        if no_act == 1:
            if len(self.onnx_node.input) == 2:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """MatrixVectorActivation needs in no
                            activation mode 2 inputs (data input and weights)"""
                )
        elif no_act == 0:
            if len(self.onnx_node.input) == 3:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """MatrixVectorActivation needs 3 inputs
                            (data input and weights and threshold values)"""
                )
        else:
            info_messages.append(
                """noActivation attribute contains {} should
                be 0 or 1""".format(
                    no_act
                )
            )
        return info_messages

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype(0):
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(0)),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        # when performing FIFO insertion on an FC layer with ext weights, the ind
        # parameter can be > 0 (referring to the weights) so handle that here
        if ind == 0:
            return DataType[self.get_nodeattr("inputDataType")]
        elif ind == 1:
            return DataType[self.get_nodeattr("weightDataType")]
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_accumulator_datatype(self):
        """Returns FINN DataType of accumulator"""
        return DataType[self.get_nodeattr("accDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.get_nodeattr("SIMD")
        elif ind == 1:
            if self.get_nodeattr("dynamic_input"):
                width = (
                    self.get_folded_input_shape(ind)[-1] * self.get_input_datatype(ind).bitwidth()
                )
            elif (
                self.get_nodeattr("mem_mode") == "internal_decoupled"
                or self.get_nodeattr("mem_mode") == "external"
            ):
                pe = self.get_nodeattr("PE")
                simd = self.get_nodeattr("SIMD")
                wp = self.get_input_datatype(1).bitwidth()
                width = pe * simd * wp
            else:
                width = 0
        elif ind == 2:
            # check if integrated thresholding and return 0
            # because threshold values are always embedded
            # or raise expection if there shouldn't be
            # a third input to the node
            act = not self.get_nodeattr("noActivation")
            if act:
                width = 0
            else:
                raise Exception("Index out of range")
        else:
            raise Exception("Index out of range")
        return width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("PE")
        return out_width

    def get_folded_input_shape(self, ind=0):
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        sf = mw // simd
        nf = mh // pe
        vecs = list(self.get_nodeattr("numInputVectors"))

        if ind == 0:
            # calculate shape of input 0
            folded_input_shape = tuple(vecs + [sf, simd])
        elif ind == 1:
            if self.get_nodeattr("dynamic_input"):
                # calculate shape of input 1 (weights dynamic)
                folded_input_shape = tuple(vecs[:2] + [mw] + [nf, pe])
            elif self.get_nodeattr("mem_mode") == "external":
                # calculate shape of input 1 (weights static and external)
                folded_input_shape = tuple(vecs + [sf * nf, simd * pe])
            else:
                raise Exception("Undefined input shape for requested input")
        else:
            raise Exception("Undefined input shape for requested input")

        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        nf = mh // pe
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_output_shape = tuple(vecs + [nf, pe])
        return folded_output_shape

    def get_normal_input_shape(self, ind=0):
        mw = self.get_nodeattr("MW")
        if ind == 0:
            vecs = list(self.get_nodeattr("numInputVectors"))
            shape = tuple(vecs + [mw])
        elif ind == 1:
            mh = self.get_nodeattr("MH")
            shape = tuple([mw, mh])
        else:
            raise Exception("Undefined input shape for requested input")
        return shape

    def get_normal_output_shape(self, ind=0):
        mh = self.get_nodeattr("MH")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_output_shape = tuple(vecs + [mh])
        return normal_output_shape

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
        wmem = mw * mh // (pe * simd)
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        if self.get_nodeattr("noActivation") == 1:
            return 0
        else:
            mh = self.get_nodeattr("MH")
            pe = self.get_nodeattr("PE")
            return mh // pe

    def uram_estimation(self):
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        omega = (D_in * D_out) / (Q * P)
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (
            (mmode == "internal_decoupled" and mstyle != "ultra")
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
        ):
            return 0
        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

    def bram_estimation(self):
        """Calculates resource estimation for BRAM based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        omega = (D_in * D_out) / (Q * P)
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (
            (mmode == "internal_decoupled" and mstyle in ["distributed", "ultra"])
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
        ):
            return 0
        # assuming SDP mode RAMB18s (see UG573 Table 1-10)
        # assuming internal_decoupled (RTL) memory,
        # which is more efficient than internal_embedded (HLS)
        if mem_width == 1:
            return math.ceil(omega / 16384)
        elif mem_width == 2:
            return math.ceil(omega / 8192)
        elif mem_width <= 4:
            return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
        elif mem_width <= 9:
            return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 9))
        elif mem_width <= 18 or omega > 512:
            return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 18))
        else:
            return (math.ceil(omega / 512)) * (math.ceil(mem_width / 36))

    def bram_efficiency_estimation(self):
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        wbits = W * D_in * D_out
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def uram_efficiency_estimation(self):
        """Function for URAM efficiency estimation: actual parameter storage
        needed divided by the allocated URAM storage (from estimation)"""
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        D_in = self.get_nodeattr("MW")
        D_out = self.get_nodeattr("MH")
        uram_est = self.uram_estimation()
        if uram_est == 0:
            return 1
        wbits = W * D_in * D_out
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    def get_exp_cycles(self):
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        num_inp_vec = self.get_nodeattr("numInputVectors")
        mh = self.get_nodeattr("MH")
        mw = self.get_nodeattr("MW")
        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        exp_cycles = (mh / pe) * (mw / simd) * np.prod(num_inp_vec) / mmv
        return int(exp_cycles)

    def minimize_accumulator_width(self, model):
        """Minimize the accumulator bit width according to the weight values,
        input data types, and size of dot product"""
        weights = model.get_initializer(self.onnx_node.input[1])
        # since in the calculation the values of the weight matrix are used,
        # for the bipolar case they need to be converted to bipolar
        if self.get_nodeattr("binaryXnorMode"):
            weights = 2 * weights - 1

        thresholds = None
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])

        idt = self.get_input_datatype(0)

        if not self.get_nodeattr("dynamic_input"):
            (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)

        # if runtime-writeable weights or dynamic input, then the values of the weights can
        # change and we need to use the worst-case values from the datatypes
        if self.get_nodeattr("runtime_writeable_weights") or self.get_nodeattr("dynamic_input"):
            mw = self.get_nodeattr("MW")
            mh = self.get_nodeattr("MH")
            wdt = self.get_input_datatype(1)
            lower_worst = wdt.min() * np.ones((mw, mh))
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_worst = wdt.max() * np.ones((mw, mh))
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(lower_range), max(upper_range))

        # if the thresholds can be used to determine range, then adjust the range
        # according to the known values of the thresholds
        if thresholds is not None:
            threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
            # set threshold datatype (and accumulator datatype implicitly)
            min_threshold = thresholds.min()
            max_threshold = thresholds.max()
            # clip threshold values
            if max_threshold > acc_max or min_threshold < acc_min:
                warnings.warn("Clipping some thresholds in %s" % self.onnx_node.name)
                thresholds = np.clip(thresholds, acc_min, acc_max)
                model.set_initializer(self.onnx_node.input[2], thresholds)
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                min_threshold = thresholds.min()
                max_threshold = thresholds.max()
            acc_min = min(min_threshold, acc_min)
            acc_max = max(max_threshold, acc_max)

        # if the acc_range is always greater than 0, then acc_max <= 2^P - 1
        if acc_min >= 0:
            acc_bit_width = np.log2(acc_max + 1)
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"UINT{acc_bit_width}"]
        # if the acc_range is signed, then acc_min >= -2^{P-1} and acc_max <=
        # 2^{P - 1} - 1, which means 2^{P - 1} >= max(-acc_min, 1 + acc_max)
        else:
            _acc_max = max(-acc_min, 1 + acc_max)
            acc_bit_width = np.log2(_acc_max) + 1
            acc_bit_width = math.ceil(acc_bit_width)
            adt = DataType[f"INT{acc_bit_width}"]

        # if activation, assert that the thresholds can be expressed with adt
        if thresholds is not None:
            assert np.vectorize(adt.allowed)(
                threshold_tensor
            ).all(), "Thresholds in %s can't be expressed with type %s" % (
                self.onnx_node.name,
                str(adt),
            )

        # if no activation, output and accumulator datatypes are the same
        if self.get_nodeattr("noActivation"):
            # if this is the last node in the graph, then ensure the datatype is
            # divisibly by 8 bits
            if model.find_direct_successors(self.onnx_node) is None:
                bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
                new_adt_name = adt.name.replace(str(adt.bitwidth()), str(bw))
                adt = DataType[new_adt_name]
            # for no-activation nodes, output dt = acc dt
            self.set_nodeattr("outputDataType", adt.name)
        self.set_nodeattr("accDataType", adt.name)
        return DataType[self.get_nodeattr("accDataType")]

    def minimize_weight_bit_width(self, model):
        """Minimize the bit width based on the values of the weights"""
        if not (
            self.get_nodeattr("runtime_writeable_weights") or self.get_nodeattr("dynamic_input")
        ):
            weights = model.get_initializer(self.onnx_node.input[1])
            w_min = weights.min()
            w_max = weights.max()
            if w_min < 0:
                if abs(w_min) > w_max:
                    wdt = DataType.get_smallest_possible(w_min)
                else:
                    wdt = DataType.get_smallest_possible(-w_max - 1)
            else:
                wdt = DataType.get_smallest_possible(w_max)
            self.set_nodeattr("weightDataType", wdt.name)
        return DataType[self.get_nodeattr("weightDataType")]

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for bipolar weights&inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        if inp_is_bipolar and wt_is_bipolar:
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
            # ensure all thresholds are integer
            assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        assert ret.shape[0] == mh, "Channels of threshold matrix are not as expected (mh)"
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        assert (
            ret.shape[0] == pe
        ), """First dimension after distribution of the
        rows between PEs is not as expected (pe)"""
        assert (
            ret.shape[1] == tmem
        ), """Second dimension after distribution of the
        rows between PEs is not as expected (tmem)"""
        assert (
            ret.shape[2] == n_thres_steps
        ), """Third dimension after distribution of the
        rows between PEs is not as expected (n_thres_steps)"""
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0 and MW % SIMD == 0
        * for bipolar {-1,+1} weights, convert to binary {0, 1}
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        mw = self.get_nodeattr("MW")
        mh = self.get_nodeattr("MH")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            mw,
            mh,
        ), """Weights matrix doesn't
        have expected shape (mw, mh)"""
        assert mw % simd == 0, "Requirement MH divisable by SIMD is violated."
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        # start by transposing the original weight matrix, since ONNX and
        # finn-hlslib use different assumptions
        # ONNX uses (in_features, out_features) and matmul(x, W)
        # finn-hlslib uses (out_features, in_features) and matmul(W, x)
        ret = orig_weight_matrix.T
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension and add a dummy outer dim
        ret = ret.reshape(1, pe, wmem, simd)
        # reverse the SIMD dimension
        ret = np.flip(ret, axis=-1)
        return ret

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib/rtllib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_input_datatype(1)
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]
        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code(weight_tensor, export_wdt, "weights", True, True)
            # write weights into C++ header file as dictated by finn-hlslib
            f_weights = open(weight_file_name, "w")
            if export_wdt.bitwidth() != 1:
                f_weights.write(
                    "const FixedPointWeights<{},{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        export_wdt.get_hls_datatype_str(),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            else:
                f_weights.write(
                    "const BinaryWeights<{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            f_weights.write(weight_hls_code)
            f_weights.close()
        elif "decoupled" in weight_file_mode:
            # create a weight stream for various flavors of internal_decoupled mode:
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
            # reverse SIMD flip for saving weights in .npy
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            # PE flip for saving weights in .dat
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)
            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                if self.get_nodeattr("dynamic_input"):
                    weight_width = weight_width * simd
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                # add zeroes to pad out file to 1024 entries
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                if self.get_nodeattr("pumpedMemory"):
                    # if pe = simd = 1, known bug, ask user to increase parallelism
                    if pe == simd == 1:
                        raise Exception(
                            """Pumped memory with pe=simd=1 is not supported.
                            Please increase parallelism."""
                        )
                    split_w_stream = np.zeros([weight_stream.shape[0] * 2], dtype=object)
                    k = 0
                    for i in range(len(weight_stream)):
                        weight = weight_stream[i]
                        split_w_stream[k] = weight[len(weight) // 2 :]
                        split_w_stream[k + 1] = weight[: len(weight) // 2]
                        k += 2
                    weight_stream = split_w_stream
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                if self.get_nodeattr("dynamic_input"):
                    weight_width = weight_width * simd
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        # split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Unknown weight_file_mode")

        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        """Generate artifacts needed by cppsim/ipgen.
        Extended: support sfcsr_mvau (spmv_sparse) by emitting 4 sparse streams.
        """
        mem_mode = self.get_nodeattr("mem_mode")
        code_gen_dir = path

        # NEW: 稀疏模式判定
        try:
            sparse_mode = self.get_nodeattr("sparse_mode")
        except AttributeError:
            sparse_mode = "dense"

        # === 1) 权值 / 稀疏流（非 runtime 动态） ===
        if not self.get_nodeattr("dynamic_input"):
            # weights, if not external
            weights = model.get_initializer(self.onnx_node.input[1])

            if mem_mode == "internal_embedded":
                # 仍按原稠密方式导出 params.h
                weight_filename = f"{code_gen_dir}/params.h"
                self.make_weight_file(weights, "hls_header", weight_filename)

            elif mem_mode in ["internal_decoupled", "external"]:
                if sparse_mode == "spmv_sparse":
                    # 稀疏：导出 4 份 npy（cppsim）+（若 internal_decoupled）4 份 .dat（ipgen）
                    self.make_spmv_files(
                        weights,
                        code_gen_dir_cppsim=code_gen_dir,
                        for_ipgen=(mem_mode == "internal_decoupled"),
                    )
                else:
                    # 稠密：保持原有逻辑
                    weight_filename_sim = f"{code_gen_dir}/input_1.npy"
                    self.make_weight_file(weights, "decoupled_npy", weight_filename_sim)
                    if mem_mode == "internal_decoupled":
                        code_gen_dir_ipgen = self.get_nodeattr("code_gen_dir_ipgen")
                        weight_filename_rtl = f"{code_gen_dir_ipgen}/memblock.dat"
                        self.make_weight_file(weights, "decoupled_verilog_dat", weight_filename_rtl)
            else:
                raise Exception(
                    'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external".'
                )

        # === 2) 阈值导出（原样保留，不变） ===
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if thresholds is not None:
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                # use UINT32 threshold export for bipolar times bipolar
                inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
                wt_is_bipolar  = self.get_input_datatype(1) == DataType["BIPOLAR"]
                # reinterpret inp/wt as bipolar if bin_xnor_mode is set
                bin_xnor_mode = False
                try:
                    bin_xnor_mode = self.get_nodeattr("binary_xnor_mode") == 1
                except Exception:
                    pass
                inp_is_bipolar = inp_is_bipolar or (self.get_input_datatype(0) == DataType["BINARY"] and bin_xnor_mode)
                wt_is_bipolar  = wt_is_bipolar  or (self.get_input_datatype(1) == DataType["BINARY"] and bin_xnor_mode)

                tdt = DataType[self.get_nodeattr("accDataType")]
                assert np.vectorize(tdt.allowed)(threshold_tensor).all(), \
                    "Thresholds in %s can't be expressed with type %s" % (self.onnx_node.name, str(tdt))

                thresholds_hls_code = numpy_to_hls_code(threshold_tensor, tdt, "thresholds", False, True)
                with open(f"{code_gen_dir}/thresh.h", "w") as f_thresh:
                    tdt_hls = tdt.get_hls_datatype_str()
                    export_odt = self.get_output_datatype()
                    if export_odt == DataType["BIPOLAR"]:
                        export_odt = DataType["BINARY"]
                    odt_hls = export_odt.get_hls_datatype_str()
                    f_thresh.write(
                        "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs = ".format(
                            self.calc_tmem(),
                            self.get_nodeattr("PE"),
                            threshold_tensor.shape[-1],
                            tdt_hls,
                            odt_hls,
                            self.get_nodeattr("ActVal"),
                            "comp::less_equal<%s, %s>" % (tdt_hls, tdt_hls),
                        )
                    )
                    f_thresh.write(thresholds_hls_code)
                    f_thresh.write(";")

    def get_op_and_param_counts(self):
        in_features = self.get_nodeattr("MW")
        out_features = self.get_nodeattr("MH")
        weight_bits = self.get_input_datatype(1).bitwidth()
        inp_bits = self.get_input_datatype(0).bitwidth()
        num_inp_vec = self.get_nodeattr("numInputVectors")
        num_repetitions = int(np.prod(num_inp_vec))
        mac_count = in_features * out_features * num_repetitions
        # cannonicalize op type: highest bitwidth operand first s.t.
        # e.g. mac_8bx4b and mac_4bx8b don't appear as two different op types
        bw1 = min(inp_bits, weight_bits)
        bw2 = max(inp_bits, weight_bits)
        mac_op_type = "op_mac_%dbx%db" % (bw1, bw2)
        weight_param_type = "param_weight_%db" % (weight_bits)
        weight_count = in_features * out_features
        ret_dict = {mac_op_type: mac_count, weight_param_type: weight_count}
        if self.get_nodeattr("noActivation") == 0:
            tdt = DataType[self.get_nodeattr("accDataType")]
            thres_bits = tdt.bitwidth()
            thres_param_type = "param_threshold_%db" % (thres_bits)
            thres_count = out_features
            ret_dict[thres_param_type] = thres_count
        return ret_dict

    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out0": []},
        }
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["internal_decoupled", "external"]:
            n_weight_inps = self.calc_wmem()
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            io_dict["inputs"]["in1"] = [0 for i in range(num_w_reps * n_weight_inps)]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        try:
            pumped_compute = self.get_nodeattr("pumpedCompute")
        except AttributeError:
            pumped_compute = 0

        if pumped_compute or self.get_nodeattr("pumpedMemory"):
            intf_names["clk2x"] = ["ap_clk2x"]

        dynamic_input = self.get_nodeattr("dynamic_input")
        mem_mode = self.get_nodeattr("mem_mode")
        if dynamic_input:
            weight_width = self.get_instream_width(1)
            weight_width = weight_width * self.get_nodeattr("SIMD")
            intf_names["s_axis"].append(("in1_V", roundup_to_integer_multiple(weight_width, 8)))
        else:
            if mem_mode == "external":
                intf_names["s_axis"].append(("in1_V", self.get_instream_width_padded(1)))
            elif mem_mode == "internal_decoupled":
                # only expose axilite interface if attribute is set
                runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
                if runtime_writeable:
                    intf_names["axilite"] = ["s_axilite"]
        return intf_names



    def make_spmv_files(self, weights, code_gen_dir_cppsim, for_ipgen=False):
        """Emit sparse-CSR (streamed) artifacts for sfcsr_mvau.

        生成 4 个 numpy（cppsim）:
            input_sfidx.npy, input_val.npy, input_mask.npy, input_rowlen.npy

        若 for_ipgen=True，同时生成 4 个 verilog .dat（ipgen）:
            memblock_sfidx.dat, memblock_val.dat, memblock_mask.dat, memblock_rowlen.dat
        """
        import math
        import numpy as np

        # --- 基本属性 ---
        MW   = int(self.get_nodeattr("MW"))
        MH   = int(self.get_nodeattr("MH"))
        PE   = int(self.get_nodeattr("PE"))
        SIMD = int(self.get_nodeattr("SIMD"))
        assert MH % PE == 0 and SIMD > 0 and MW > 0, "Require MH%PE==0 and valid MW/SIMD."

        # 权值位宽（BIPOLAR -> BINARY）
        wdt = self.get_input_datatype(1)
        export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
        wbits = export_wdt.bitwidth()

        # sf 索引位宽
        ncols_per_lane = (MW + SIMD - 1) // SIMD  # ceil(MW/SIMD)
        sfidx_width = max(1, int(math.ceil(math.log2(ncols_per_lane))))
        sfdt     = DataType["UINT{}".format(sfidx_width)] if sfidx_width <= 32 else DataType["UINT32"]
        maskdt   = DataType["BINARY"]
        rowlendt = DataType["UINT16"]

        # --- (MW, MH) -> (MH, MW) -> interleave -> (PE, TMEM, MW) ---
        assert tuple(weights.shape) == (MW, MH), "Weight tensor shape must be (MW, MH)."
        W = np.asarray(weights.T)  # (MH, MW)
        if wdt == DataType["BIPOLAR"]:
            W = (W + 1) / 2.0
        # 分配到 PEs（注意：返回 (PE, TMEM, MW)）
        W_int = interleave_matrix_outer_dim_from_partitions(W, PE)
        W_int = np.asarray(W_int, dtype=np.float32)
        assert W_int.ndim == 3 and W_int.shape[0] == PE and W_int.shape[2] == MW, \
            f"interleave result shape must be (PE, TMEM, MW); got {W_int.shape}"
        TMEM = W_int.shape[1]

        # --- 展开时间步（每个 g=0..TMEM-1 是一个“组”，包含 PE 行）---
        sfidx_words, val_words, mask_words, rowlen_words = [], [], [], []
        lane_idx = np.arange(MW, dtype=np.int32)   # 0..MW-1

        for g in range(TMEM):
            # 为该“组”预先收集每行的 (按 lane) 非零列/值
            lane_cols_per_row = []
            lane_vals_per_row = []
            rowlens = []

            for p in range(PE):
                row = W_int[p, g, :].reshape(-1)                 # (MW,)
                nz_mask = row != 0.0
                cols_per_lane, vals_per_lane = [], []
                for lane in range(SIMD):
                    sel_lane = (lane_idx % SIMD) == lane
                    chosen = sel_lane & nz_mask
                    cols = lane_idx[chosen].tolist()             # 自然顺序
                    vals = row[chosen].tolist()
                    cols_per_lane.append(cols)
                    vals_per_lane.append(vals)
                lane_cols_per_row.append(cols_per_lane)
                lane_vals_per_row.append(vals_per_lane)
                rowlens.append(max(len(cs) for cs in cols_per_lane) if SIMD > 0 else 0)

            group_T = max(rowlens) if rowlens else 0

            # 在该组内展开 group_T 个拍：第 t 拍取每行每个 lane 的第 t 个元素
            for t in range(group_T):
                sf_line, val_line, m_line, rl_line = [], [], [], []
                for p in range(PE):
                    rl_line.append(rowlens[p])  # 每拍附带该行的 rowlen
                    for lane in range(SIMD):
                        cols = lane_cols_per_row[p][lane]
                        vals = lane_vals_per_row[p][lane]
                        if t < len(cols):
                            c = int(cols[t])
                            v = float(vals[t])
                            sf_line.append(c // SIMD)
                            val_line.append(v)
                            m_line.append(1)
                        else:
                            sf_line.append(0)
                            val_line.append(0.0)
                            m_line.append(0)

                # 追加一拍（合并 PE×SIMD 槽位）
                sfidx_words.append(np.array(sf_line, dtype=np.float32))
                val_words.append(np.array(val_line, dtype=np.float32))
                mask_words.append(np.array(m_line, dtype=np.float32))
                rowlen_words.append(np.array(rl_line, dtype=np.float32))

        # 极端稀疏：确保至少 1 拍
        if len(sfidx_words) == 0:
            sfidx_words  = [np.zeros((PE * SIMD,), dtype=np.float32)]
            val_words    = [np.zeros((PE * SIMD,), dtype=np.float32)]
            mask_words   = [np.zeros((PE * SIMD,), dtype=np.float32)]
            rowlen_words = [np.zeros((PE,),        dtype=np.float32)]

        # 拼装成 (1, T, *)（FINN 的 npy2apintstream 约定）
        T = len(sfidx_words)
        sfidx_arr  = np.stack(sfidx_words,  axis=0).reshape(1, T, PE * SIMD)
        val_arr    = np.stack(val_words,    axis=0).reshape(1, T, PE * SIMD)
        mask_arr   = np.stack(mask_words,   axis=0).reshape(1, T, PE * SIMD)
        rowlen_arr = np.stack(rowlen_words, axis=0).reshape(1, T, PE)

        # --- 保存给 cppsim ---
        np.save(f"{code_gen_dir_cppsim}/input_sfidx.npy",  sfidx_arr)
        np.save(f"{code_gen_dir_cppsim}/input_val.npy",    val_arr)
        np.save(f"{code_gen_dir_cppsim}/input_mask.npy",   mask_arr)
        np.save(f"{code_gen_dir_cppsim}/input_rowlen.npy", rowlen_arr)

        # --- 若需要，保存 .dat 给 ipgen 的 memstream ---
        if for_ipgen:
            code_gen_dir_ipgen = self.get_nodeattr("code_gen_dir_ipgen")

            def pack_to_hex(arr3d, elem_dt, word_width_bits):
                word_width_padded = roundup_to_integer_multiple(word_width_bits, 4)
                packed = pack_innermost_dim_as_hex_string(
                    arr3d.astype(np.float32), elem_dt, word_width_padded, prefix=""
                )
                return packed.flatten().copy()

            sf_hex = pack_to_hex(sfidx_arr,  sfdt,        PE * SIMD * sfidx_width)
            v_hex  = pack_to_hex(val_arr,    export_wdt,  PE * SIMD * wbits)
            m_hex  = pack_to_hex(mask_arr,   maskdt,      PE * SIMD * 1)
            rl_hex = pack_to_hex(rowlen_arr, rowlendt,    PE * 16)

            # 可选：2x pumped memory（把每行十六进制串平分并上下对调）
            def pump2x(hex_stream):
                if not self.get_nodeattr("pumpedMemory"):
                    return hex_stream
                out = np.empty((hex_stream.shape[0] * 2,), dtype=object)
                k = 0
                for s in hex_stream:
                    mid = len(s) // 2
                    out[k]   = s[mid:]
                    out[k+1] = s[:mid]
                    k += 2
                return out

            sf_hex = pump2x(sf_hex)
            v_hex  = pump2x(v_hex)
            m_hex  = pump2x(m_hex)
            rl_hex = pump2x(rl_hex)

            with open(f"{code_gen_dir_ipgen}/memblock_sfidx.dat", "w") as f:
                for s in sf_hex: f.write(s + "\n")
            with open(f"{code_gen_dir_ipgen}/memblock_val.dat", "w") as f:
                for s in v_hex:  f.write(s + "\n")
            with open(f"{code_gen_dir_ipgen}/memblock_mask.dat", "w") as f:
                for s in m_hex:  f.write(s + "\n")
            with open(f"{code_gen_dir_ipgen}/memblock_rowlen.dat", "w") as f:
                for s in rl_hex: f.write(s + "\n")


    # def code_generation_ipi(self):
    #     source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
    #     cmd = ["file mkdir %s" % source_target]
    #     dyn_input = self.get_nodeattr("dynamic_input")
    #     mem_mode = self.get_nodeattr("mem_mode")
    #     sname = "V"

    #     # check if additional components are needed
    #     if dyn_input or mem_mode == "internal_decoupled":
    #         node_name = self.onnx_node.name
    #         # create a hierarchy for this layer, with the same port names
    #         clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
    #         rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
    #         dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
    #         din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
    #         cmd.append("create_bd_cell -type hier %s" % node_name)
    #         # clock and reset
    #         cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
    #         cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
    #         # if we need a 2x clock for either compute or memory, instantiate the 2x clk port
    #         try:
    #             pumped_compute = self.get_nodeattr("pumpedCompute")
    #         except AttributeError:
    #             pumped_compute = 0
    #         if pumped_compute or self.get_nodeattr("pumpedMemory"):
    #             clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
    #             cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk2x_name))
    #         else:
    #             clk2x_name = None
    #         # streams
    #         cmd.append(
    #             "create_bd_intf_pin -mode Master "
    #             "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
    #         )
    #         cmd.append(
    #             "create_bd_intf_pin -mode Slave "
    #             "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
    #         )
    #         # instantiate the RTL block
    #         # Instantiate either the HLS or RTL IP depending on operator
    #         self.instantiate_ip(cmd)
    #         # connect MVAU
    #         cmd.append(
    #             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
    #             % (node_name, rst_name, node_name, node_name, rst_name)
    #         )
    #         cmd.append(
    #             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
    #             % (node_name, clk_name, node_name, node_name, clk_name)
    #         )
    #         cmd.append(
    #             "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #             "[get_bd_intf_pins %s/%s/%s]"
    #             % (node_name, din_name, node_name, node_name, din_name)
    #         )
    #         cmd.append(
    #             "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #             "[get_bd_intf_pins %s/%s/%s]"
    #             % (node_name, dout_name, node_name, node_name, dout_name)
    #         )

    #         code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
    #         # EARLY sparse-mode handling: skip dense memstream template discovery entirely
    #         try:
    #             sparse_mode = self.get_nodeattr("sparse_mode")
    #         except Exception:
    #             sparse_mode = "dense"
    #         if sparse_mode == "spmv_sparse" and (not dyn_input):
    #             set_suffixes = ["sfidx", "val", "mask", "rowlen"]
    #             # Discover wrapper templates for each suffix within code_gen_dir
    #             strm_tmpl_map = {}
    #             for fname in os.listdir(code_gen_dir):
    #                 for sfx in set_suffixes:
    #                     if fname.endswith(f"_memstream_wrapper_{sfx}.v"):
    #                         strm_tmpl_map[sfx] = fname

    #             missing = [sfx for sfx in set_suffixes if sfx not in strm_tmpl_map]
    #             if missing:
    #                 raise Exception(
    #                     f"Missing sparse memstream wrappers for: {missing}. "
    #                     f"Expect *_memstream_wrapper_{{sfidx,val,mask,rowlen}}.v in {code_gen_dir}"
    #                 )

    #             # Use same rtllib directories as dense path
    #             axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
    #             ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")

    #             for sfx in set_suffixes:
    #                 strm_tmpl_sfx = strm_tmpl_map[sfx]
    #                 strm_tmpl_name_sfx = strm_tmpl_sfx[:-2]

    #                 # add required Verilog sources for this wrapper
    #                 for f in [
    #                     os.path.join(code_gen_dir, strm_tmpl_sfx),
    #                     axi_dir + "axilite.sv",
    #                     ms_rtllib_dir + "memstream_axi.sv",
    #                     ms_rtllib_dir + "memstream.sv",
    #                 ]:
    #                     cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]

    #                 # instantiate wrapper as a hier cell
    #                 strm_inst_sfx = node_name + f"_wstrm_{sfx}"
    #                 cmd.append(
    #                     "create_bd_cell -type hier -reference %s /%s/%s"
    #                     % (strm_tmpl_name_sfx, node_name, strm_inst_sfx)
    #                 )

    #                 # connect clocks and reset
    #                 cmd.append(
    #                     "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
    #                     % (node_name, clk_name, node_name, strm_inst_sfx)
    #                 )
    #                 cmd.append(
    #                     "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
    #                     % (node_name, rst_name, node_name, strm_inst_sfx)
    #                 )

    #                 # if using 2x pumped memory, connect memstreamer's 2x clk to 2x clock port
    #                 # otherwise connect it to the regular clock port.
    #                 if self.get_nodeattr("pumpedMemory"):
    #                     cmd.append(
    #                         "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                         % (node_name, clk2x_name, node_name, strm_inst_sfx)
    #                     )
    #                 else:
    #                     cmd.append(
    #                         "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                         % (node_name, clk_name, node_name, strm_inst_sfx)
    #                     )

    #                 # connect m_axis to HLS op's corresponding sparse input
    #                 cmd.append(
    #                     "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
    #                     "[get_bd_intf_pins %s/%s/in1_%s_%s]"
    #                     % (node_name, strm_inst_sfx, node_name, node_name, sfx, sname)
    #                 )

    #             # Expose 4 AXI-Lite ports to load memblock_{sfx}.dat
    #             for sfx in set_suffixes:
    #                 axilite_name_sfx = f"s_axilite_{sfx}"
    #                 strm_inst_sfx = node_name + f"_wstrm_{sfx}"
    #                 cmd.append(
    #                     "create_bd_intf_pin -mode Slave "
    #                     "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
    #                     % (node_name, axilite_name_sfx)
    #                 )
    #                 cmd.append(
    #                     "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #                     "[get_bd_intf_pins %s/%s/s_axilite]"
    #                     % (node_name, axilite_name_sfx, node_name, strm_inst_sfx)
    #                 )

    #             cmd.append("assign_bd_address")
    #             return cmd


    #         if dyn_input:
    #             # dynamic loader
    #             dynld_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/dynload/hdl/")
    #             file_suffix = "_dynamic_load_wrapper.v"
    #             # automatically find memstream verilog component in code generation directory
    #             for fname in os.listdir(code_gen_dir):
    #                 if fname.endswith(file_suffix):
    #                     dynld_tmpl = fname
    #             dynld_tmpl_name = dynld_tmpl[:-2]
    #             sourcefiles = [
    #                 os.path.join(code_gen_dir, dynld_tmpl),
    #                 dynld_rtllib_dir + "ram_p_c.sv",
    #                 dynld_rtllib_dir + "dynamic_load.sv",
    #             ]
    #             for f in sourcefiles:
    #                 cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
    #             dynld_inst = node_name + "_wdynld"
    #             # instantiate the cell
    #             cmd.append(
    #                 "create_bd_cell -type hier -reference %s /%s/%s"
    #                 % (dynld_tmpl_name, node_name, dynld_inst)
    #             )
    #             # additional dynamic input
    #             win_name = self.get_verilog_top_module_intf_names()["s_axis"][1][0]
    #             cmd.append(
    #                 "create_bd_intf_pin -mode Slave "
    #                 "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, win_name)
    #             )
    #             # connect
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
    #                 % (node_name, clk_name, node_name, dynld_inst)
    #             )
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
    #                 % (node_name, rst_name, node_name, dynld_inst)
    #             )
    #             cmd.append(
    #                 "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
    #                 "[get_bd_intf_pins %s/%s/in1_%s]"
    #                 % (node_name, dynld_inst, node_name, node_name, sname)
    #             )
    #             cmd.append(
    #                 "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #                 "[get_bd_intf_pins %s/%s/s_axis_0]"
    #                 % (node_name, win_name, node_name, dynld_inst)
    #             )
    #         else:
    #             # memstream
    #             runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
    #             axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
    #             ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
    #             file_suffix = "_memstream_wrapper.v"
    #             # automatically find memstream verilog component in code generation directory
    #             for fname in os.listdir(code_gen_dir):
    #                 if fname.endswith(file_suffix):
    #                     strm_tmpl = fname
    #             strm_tmpl_name = strm_tmpl[:-2]
    #             sourcefiles = [
    #                 os.path.join(code_gen_dir, strm_tmpl),
    #                 axi_dir + "axilite.sv",
    #                 ms_rtllib_dir + "memstream_axi.sv",
    #                 ms_rtllib_dir + "memstream.sv",
    #             ]

    #             for f in sourcefiles:
    #                 cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
    #             strm_inst = node_name + "_wstrm"
    #             # instantiate the cell
    #             cmd.append(
    #                 "create_bd_cell -type hier -reference %s /%s/%s"
    #                 % (strm_tmpl_name, node_name, strm_inst)
    #             )
    #             # connect
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
    #                 % (node_name, clk_name, node_name, strm_inst)
    #             )
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
    #                 % (node_name, rst_name, node_name, strm_inst)
    #             )
    #             # if using 2x pumped memory, connect the memstreamer's 2x clk input
    #             # to the 2x clock port. otherwise connect it to the regular clock port.
    #             if self.get_nodeattr("pumpedMemory"):
    #                 cmd.append(
    #                     "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                     % (node_name, clk2x_name, node_name, strm_inst)
    #                 )
    #             else:
    #                 cmd.append(
    #                     "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                     % (node_name, clk_name, node_name, strm_inst)
    #                 )
    #             cmd.append(
    #                 "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
    #                 "[get_bd_intf_pins %s/%s/in1_%s]"
    #                 % (node_name, strm_inst, node_name, node_name, sname)
    #             )
    #             # runtime writeable weights
    #             if runtime_writable:
    #                 axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
    #                 cmd.append(
    #                     "create_bd_intf_pin -mode Slave "
    #                     "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
    #                     % (node_name, axilite_name)
    #                 )
    #                 cmd.append(
    #                     "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #                     "[get_bd_intf_pins %s/%s/%s]"
    #                     % (node_name, axilite_name, node_name, strm_inst, axilite_name)
    #                 )
    #                 # TODO calculate and pass in segment size here
    #                 cmd.append("assign_bd_address")

    #         # save bd
    #         cmd.append("save_bd_design")
    #     elif mem_mode == "internal_embedded" or mem_mode == "external":
    #         # base class impl sufficient for internal_embedded/external modes
    #         self.instantiate_ip(cmd)
    #     else:
    #         raise Exception("Unrecognized mem_mode for MatrixVectorActivation")
    #     return cmd

    # def code_generation_ipi(self):
    #     """
    #     生成本层的 IPI 连接（Vivado Tcl 命令列表）。
    #     - 稀疏层(sparse_mode == "spmv_sparse")：4 路 memstream，连到 sfidx_V/val_V/mask_V/rowlen_V
    #     - 稠密层(默认)：1 路 memstream，连到 in1_V
    #     仅兼容稀疏 wrapper 新命名：<node>_{sfidx|val|mask|rowlen}_memstream_wrapper.v（模块名=文件名去掉 .v）
    #     """
    #     import os

    #     cmd = []

    #     node_name     = self.onnx_node.name
    #     source_target = "./ip/verilog/rtl_ops/%s" % node_name
    #     code_gen_dir  = self.get_nodeattr("code_gen_dir_ipgen")

    #     # mkdir for sources
    #     cmd.append("file mkdir %s" % source_target)

    #     # FINN-rtllib 路径
    #     axi_dir       = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
    #     ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
    #     dynld_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/dynload/hdl/")

    #     # 属性
    #     dyn_input  = bool(self.get_nodeattr("dynamic_input"))
    #     mem_mode   = self.get_nodeattr("mem_mode")
    #     pumped_mem = bool(self.get_nodeattr("pumpedMemory"))
    #     try:
    #         sparse_mode = self.get_nodeattr("sparse_mode")
    #     except Exception:
    #         sparse_mode = "dense"

    #     # ===== 统一：先创建父层级 /<node> 并暴露顶层端口、再实例化 HLS IP /<node>/<node> =====
    #     # 拿到 HLS 顶层 intf 名称（与 FINN 旧逻辑一致）
    #     vnames = self.get_verilog_top_module_intf_names()
    #     clk_name = vnames["clk"][0]          # "ap_clk"
    #     rst_name = vnames["rst"][0]          # "ap_rst_n"
    #     dout_name = vnames["m_axis"][0][0]   # "out0_V"
    #     din_name  = vnames["s_axis"][0][0]   # "in0_V"

    #     # 创建父层级与顶层引脚/接口
    #     cmd.append("create_bd_cell -type hier %s" % node_name)
    #     cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
    #     cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))

    #     # 如果需要 2x 时钟，暴露 ap_clk2x
    #     try:
    #         pumped_compute = self.get_nodeattr("pumpedCompute")
    #     except AttributeError:
    #         pumped_compute = 0
    #     if pumped_compute or pumped_mem:
    #         clk2x_name = vnames["clk2x"][0]  # "ap_clk2x"
    #         cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk2x_name))
    #     else:
    #         clk2x_name = None

    #     # 父层级暴露 in0/out0 AXIS
    #     cmd.append(
    #         "create_bd_intf_pin -mode Master "
    #         "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
    #     )
    #     cmd.append(
    #         "create_bd_intf_pin -mode Slave "
    #         "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
    #     )

    #     # 实例化 HLS 或 RTL IP（放在 /<node>/<node>）
    #     self.instantiate_ip(cmd)

    #     # 连接 HLS 与父层级基础端口/接口
    #     cmd.append(
    #         "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
    #         % (node_name, rst_name, node_name, node_name, rst_name)
    #     )
    #     cmd.append(
    #         "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
    #         % (node_name, clk_name, node_name, node_name, clk_name)
    #     )
    #     cmd.append(
    #         "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #         "[get_bd_intf_pins %s/%s/%s]"
    #         % (node_name, din_name, node_name, node_name, din_name)
    #     )
    #     cmd.append(
    #         "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #         "[get_bd_intf_pins %s/%s/%s]"
    #         % (node_name, dout_name, node_name, node_name, dout_name)
    #     )

    #     # ===== 稀疏模式（只兼容新命名；且仅在非动态权重时才接 4 路 wrapper） =====
    #     if (sparse_mode == "spmv_sparse") and (not dyn_input):
    #         set_suffixes = ["sfidx", "val", "mask", "rowlen"]

    #         # 校验 4 个 wrapper 存在；模块名=文件名去掉 .v
    #         for sfx in set_suffixes:
    #             vfile = f"{node_name}_{sfx}_memstream_wrapper.v"
    #             vpath = os.path.join(code_gen_dir, vfile)
    #             if not os.path.exists(vpath):
    #                 raise Exception(
    #                     f"[SPMV-IPI] Missing wrapper {vpath}. "
    #                     f"Only NEW naming supported. Please run generate_hdl_memstream_spmv() first."
    #                 )

    #         for sfx in set_suffixes:
    #             vfile   = f"{node_name}_{sfx}_memstream_wrapper.v"
    #             modref  = vfile[:-2]
    #             inst    = f"{node_name}_wstrm_{sfx}"

    #             # add_files
    #             for f in [
    #                 os.path.join(code_gen_dir, vfile),
    #                 axi_dir + "axilite.sv",
    #                 ms_rtllib_dir + "memstream_axi.sv",
    #                 ms_rtllib_dir + "memstream.sv",
    #             ]:
    #                 cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
    #             cmd.append("update_compile_order -fileset sources_1")

    #             # 实例化 wrapper 到 /<node>/<inst>
    #             cmd.append(
    #                 "create_bd_cell -type hier -reference %s /%s/%s"
    #                 % (modref, node_name, inst)
    #             )

    #             # clocks & reset
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
    #                 % (node_name, clk_name, node_name, inst)
    #             )
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
    #                 % (node_name, rst_name, node_name, inst)
    #             )
    #             if pumped_mem:
    #                 # 有 2x：接父 ap_clk2x
    #                 cmd.append(
    #                     "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                     % (node_name, clk2x_name, node_name, inst)
    #                 )
    #             else:
    #                 # 无 2x：把 ap_clk2x 绑到 ap_clk
    #                 cmd.append(
    #                     "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                     % (node_name, clk_name, node_name, inst)
    #                 )

    #             # AXIS：wrapper -> HLS <sfx>_V   （注意：没有 in1_ 前缀）
    #             cmd.append(
    #                 "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
    #                 "[get_bd_intf_pins %s/%s/%s_V]"
    #                 % (node_name, inst, node_name, node_name, sfx)
    #             )

    #         # 暴露 4 个 AXI-Lite 口到父层级，并接各 wrapper 的 s_axilite
    #         for sfx in set_suffixes:
    #             inst = f"{node_name}_wstrm_{sfx}"
    #             axil = f"s_axilite_{sfx}"
    #             cmd.append(
    #                 "create_bd_intf_pin -mode Slave "
    #                 "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
    #                 % (node_name, axil)
    #             )
    #             cmd.append(
    #                 "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #                 "[get_bd_intf_pins %s/%s/s_axilite]"
    #                 % (node_name, axil, node_name, inst)
    #             )

    #         cmd.append("assign_bd_address")
    #         cmd.append("save_bd_design")
    #         return cmd

    #     # ===== 动态权重输入（保持 FINN 原有动态加载路径） =====
    #     if dyn_input:
    #         # *_dynamic_load_wrapper.v
    #         file_suffix = "_dynamic_load_wrapper.v"
    #         dynld_tmpl = None
    #         for fname in os.listdir(code_gen_dir):
    #             if fname.endswith(file_suffix):
    #                 dynld_tmpl = fname
    #                 break
    #         if dynld_tmpl is None:
    #             raise Exception(
    #                 f"[DYNLOAD-IPI] Missing *{file_suffix} under {code_gen_dir}"
    #             )
    #         dynld_tmpl_name = dynld_tmpl[:-2]
    #         sourcefiles = [
    #             os.path.join(code_gen_dir, dynld_tmpl),
    #             dynld_rtllib_dir + "ram_p_c.sv",
    #             dynld_rtllib_dir + "dynamic_load.sv",
    #         ]
    #         for f in sourcefiles:
    #             cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
    #         cmd.append("update_compile_order -fileset sources_1")

    #         dynld_inst = node_name + "_wdynld"
    #         cmd.append(
    #             "create_bd_cell -type hier -reference %s /%s/%s"
    #             % (dynld_tmpl_name, node_name, dynld_inst)
    #         )
    #         # 额外暴露动态权重入口（第二路 s_axis）
    #         win_name = self.get_verilog_top_module_intf_names()["s_axis"][1][0]
    #         cmd.append(
    #             "create_bd_intf_pin -mode Slave "
    #             "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, win_name)
    #         )
    #         # clocks & reset
    #         cmd.append(
    #             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
    #             % (node_name, clk_name, node_name, dynld_inst)
    #         )
    #         cmd.append(
    #             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
    #             % (node_name, rst_name, node_name, dynld_inst)
    #         )
    #         # AXIS：dynloader -> HLS in1_V
    #         cmd.append(
    #             "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
    #             "[get_bd_intf_pins %s/%s/in1_V]"
    #             % (node_name, dynld_inst, node_name, node_name)
    #         )
    #         # AXIS：上游 -> dynloader
    #         cmd.append(
    #             "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #             "[get_bd_intf_pins %s/%s/s_axis_0]"
    #             % (node_name, win_name, node_name, dynld_inst)
    #         )

    #         cmd.append("save_bd_design")
    #         return cmd

    #     # ===== 稠密（单路 memstream -> in1_V） =====
    #     # 仅在 internal_decoupled / external 有意义；embedded 没有外部流
    #     if mem_mode in ["internal_decoupled", "external"]:
    #         runtime_writable = (self.get_nodeattr("runtime_writeable_weights") == 1)

    #         # 找到唯一的 *_memstream_wrapper.v
    #         strm_tmpl = None
    #         candidates = [f for f in os.listdir(code_gen_dir) if f.endswith("_memstream_wrapper.v")]
    #         if len(candidates) == 1:
    #             strm_tmpl = candidates[0]
    #         elif len(candidates) == 0:
    #             raise Exception(f"[DENSE-IPI] No '*_memstream_wrapper.v' under {code_gen_dir}")
    #         else:
    #             raise Exception(f"[DENSE-IPI] Expect exactly one '*_memstream_wrapper.v' under {code_gen_dir}, got {len(candidates)}")

    #         strm_modref = strm_tmpl[:-2]
    #         strm_inst   = node_name + "_wstrm"

    #         # add_files
    #         for f in [
    #             os.path.join(code_gen_dir, strm_tmpl),
    #             axi_dir + "axilite.sv",
    #             ms_rtllib_dir + "memstream_axi.sv",
    #             ms_rtllib_dir + "memstream.sv",
    #         ]:
    #             cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
    #         cmd.append("update_compile_order -fileset sources_1")

    #         # 实例化 wrapper
    #         cmd.append(
    #             "create_bd_cell -type hier -reference %s /%s/%s"
    #             % (strm_modref, node_name, strm_inst)
    #         )

    #         # clocks & reset
    #         cmd.append(
    #             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
    #             % (node_name, clk_name, node_name, strm_inst)
    #         )
    #         cmd.append(
    #             "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
    #             % (node_name, rst_name, node_name, strm_inst)
    #         )
    #         if pumped_mem:
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                 % (node_name, clk2x_name, node_name, strm_inst)
    #             )
    #         else:
    #             cmd.append(
    #                 "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
    #                 % (node_name, clk_name, node_name, strm_inst)
    #             )

    #         # AXIS：wrapper -> HLS in1_V
    #         cmd.append(
    #             "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
    #             "[get_bd_intf_pins %s/%s/in1_V]"
    #             % (node_name, strm_inst, node_name, node_name)
    #         )

    #         # 仅 runtime 可写时暴露 AXI-Lite
    #         if runtime_writable:
    #             axilite_name = "s_axilite"
    #             cmd.append(
    #                 "create_bd_intf_pin -mode Slave "
    #                 "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
    #                 % (node_name, axilite_name)
    #             )
    #             cmd.append(
    #                 "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
    #                 "[get_bd_intf_pins %s/%s/%s]"
    #                 % (node_name, axilite_name, node_name, strm_inst, axilite_name)
    #             )
    #             cmd.append("assign_bd_address")

    #         cmd.append("save_bd_design")
    #         return cmd

    #     # internal_embedded 或其他情况：只放 HLS/RTL（已在前面 instantiate_ip 并完成基础连线）
    #     cmd.append("save_bd_design")
    #     return cmd


    
    # def code_generation_ipi(self):
    #     """
    #     生成本层 IPI 连接（Vivado Tcl 命令列表）。
    #     - 先确保 /<node> 父层级与 /<node>/<node> HLS IP 存在，并完成基础连线
    #     - 稀疏(sparse_mode=="spmv_sparse")：4 路 memstream -> <sfx>_V （仅新命名）
    #     - 稠密(默认)：单 memstream -> in1_V
    #     - 动态权重(dynamic_input)：使用 *_dynamic_load_wrapper.v
    #     说明：
    #     稀疏 wrapper 仅兼容新命名：<node>_{sfidx|val|mask|rowlen}_memstream_wrapper.v（模块名=文件名去 .v）
    #     """
    #     import os

    #     cmd = []

    #     FINN_ROOT = os.environ.get("FINN_ROOT", "")
    #     node_name     = self.onnx_node.name
    #     parent        = f"/{node_name}"
    #     hlsop         = f"/{node_name}/{node_name}"
    #     source_target = f"./ip/verilog/rtl_ops/{node_name}"
    #     code_gen_dir  = self.get_nodeattr("code_gen_dir_ipgen")

    #     # rtllib 路径
    #     axi_dir          = os.path.join(FINN_ROOT, "finn-rtllib/axi/hdl/")
    #     ms_rtllib_dir    = os.path.join(FINN_ROOT, "finn-rtllib/memstream/hdl/")
    #     dynld_rtllib_dir = os.path.join(FINN_ROOT, "finn-rtllib/dynload/hdl/")

    #     # 属性
    #     dyn_input   = bool(self.get_nodeattr("dynamic_input"))
    #     mem_mode    = self.get_nodeattr("mem_mode")
    #     pumped_mem  = bool(self.get_nodeattr("pumpedMemory"))
    #     try:
    #         sparse_mode = self.get_nodeattr("sparse_mode")
    #     except Exception:
    #         sparse_mode = "dense"

    #     # 端口名（从 HLS 顶层抽取，避免硬编码）
    #     vnames      = self.get_verilog_top_module_intf_names()
    #     clk_name    = vnames["clk"][0]                 # 通常 "ap_clk"
    #     rst_name    = vnames["rst"][0]                 # 通常 "ap_rst_n"
    #     din_name    = vnames["s_axis"][0][0]           # 通常 "in0_V"
    #     dout_name   = vnames["m_axis"][0][0]           # 通常 "out0_V"

    #     # ---- 打开/创建 BD + 父层级 + HLS IP，并完成基础连线 ----
    #     cmd += [
    #         # 确保 BD 打开
    #         'if {[catch {current_bd_design} _cur_bd]} { set _cur_bd "" }',
    #         'if {$_cur_bd eq ""} { create_bd_design "finn_design" }',

    #         # 目标目录
    #         f'file mkdir {source_target}',

    #         # 父层级 /<node>（若无则创建）与其顶层 pin/接口（幂等创建）
    #         f'if {{![llength [get_bd_cells -quiet {parent}]]}} {{ create_bd_cell -type hier {node_name} }}',
    #         f'if {{![llength [get_bd_pins -quiet {parent}/{clk_name}]]}} {{ create_bd_pin -dir I -type clk {parent}/{clk_name} }}',
    #         f'if {{![llength [get_bd_pins -quiet {parent}/{rst_name}]]}} {{ create_bd_pin -dir I -type rst {parent}/{rst_name} }}',
    #         # ap_clk2x：仅在 pumped_mem 时暴露
    #         (f'if {{![llength [get_bd_pins -quiet {parent}/ap_clk2x]]}} {{ create_bd_pin -dir I -type clk {parent}/ap_clk2x }}'
    #         if pumped_mem else '# no ap_clk2x on parent'),

    #         f'if {{![llength [get_bd_intf_pins -quiet {parent}/{dout_name}]]}} {{ '
    #         f'  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 {parent}/{dout_name} }}',
    #         f'if {{![llength [get_bd_intf_pins -quiet {parent}/{din_name}]]}} {{ '
    #         f'  create_bd_intf_pin -mode Slave  -vlnv xilinx.com:interface:axis_rtl:1.0 {parent}/{din_name} }}',

    #         # HLS IP /<node>/<node>（若无则从 IP Catalog 实例化 xilinx.com:hls:<node>:*）
    #         f'if {{![llength [get_bd_cells -quiet {hlsop}]]}} {{ '
    #         f'  set _defs [get_ipdefs -all -filter "VLNV =~ xilinx.com:hls:{node_name}:*"]; '
    #         f'  if {{[llength $_defs]}} {{ create_bd_cell -type ip -vlnv [lindex $_defs 0] {hlsop} }} '
    #         f'  else {{ error {{HLS IP xilinx.com:hls:{node_name}:* not found}} }} '
    #         f'}}',

    #         # 连接父层级 与 HLS 顶层端口/接口
    #         f'if {{[llength [get_bd_pins -quiet {hlsop}/{rst_name}]]}} '
    #         f'{{ connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {hlsop}/{rst_name}] }}',
    #         f'if {{[llength [get_bd_pins -quiet {hlsop}/{clk_name}]]}} '
    #         f'{{ connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {hlsop}/{clk_name}] }}',
    #         f'if {{[llength [get_bd_intf_pins -quiet {hlsop}/{din_name}]]}} '
    #         f'{{ connect_bd_intf_net [get_bd_intf_pins {parent}/{din_name}] [get_bd_intf_pins {hlsop}/{din_name}] }}',
    #         f'if {{[llength [get_bd_intf_pins -quiet {hlsop}/{dout_name}]]}} '
    #         f'{{ connect_bd_intf_net [get_bd_intf_pins {parent}/{dout_name}] [get_bd_intf_pins {hlsop}/{dout_name}] }}',
    #     ]

    #     # =============== 稀疏（仅新命名） ===============
    #     if (sparse_mode == "spmv_sparse") and (not dyn_input):
    #         set_suffixes = ["sfidx", "val", "mask", "rowlen"]

    #         # 校验 4 个 wrapper 文件存在；模块名=文件名去 .v
    #         for sfx in set_suffixes:
    #             vfile = f"{node_name}_{sfx}_memstream_wrapper.v"
    #             vpath = os.path.join(code_gen_dir, vfile)
    #             if not os.path.exists(vpath):
    #                 raise Exception(
    #                     f"[SPMV-IPI] Missing {vpath}. Only NEW naming supported. "
    #                     f"Please run generate_hdl_memstream_spmv() first."
    #                 )

    #         for sfx in set_suffixes:
    #             vfile  = f"{node_name}_{sfx}_memstream_wrapper.v"
    #             modref = vfile[:-2]                   # module == filename without ".v"
    #             inst   = f"{node_name}_wstrm_{sfx}"

    #             # add_files（带 -force）+ 刷新编译顺序
    #             for f in [
    #                 os.path.join(code_gen_dir, vfile),
    #                 os.path.join(axi_dir, "axilite.sv"),
    #                 os.path.join(ms_rtllib_dir, "memstream_axi.sv"),
    #                 os.path.join(ms_rtllib_dir, "memstream.sv"),
    #             ]:
    #                 cmd += [f"add_files -copy_to {source_target} -force -norecurse {f}"]
    #             cmd.append("update_compile_order -fileset sources_1")

    #             # 实例化 wrapper
    #             cmd.append(f"create_bd_cell -type hier -reference {modref} {parent}/{inst}")

    #             # clocks & reset
    #             cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{inst}/ap_clk]")
    #             cmd.append(f"connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {parent}/{inst}/ap_rst_n]")
    #             if pumped_mem:
    #                 cmd.append(f"connect_bd_net [get_bd_pins {parent}/ap_clk2x] [get_bd_pins {parent}/{inst}/ap_clk2x]")
    #             else:
    #                 cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{inst}/ap_clk2x]")

    #             # AXIS：wrapper -> HLS <sfx>_V
    #             cmd.append(
    #                 f"connect_bd_intf_net [get_bd_intf_pins {parent}/{inst}/m_axis_0] "
    #                 f"[get_bd_intf_pins {hlsop}/{sfx}_V]"
    #             )

    #         # 暴露 4 个 AXI-Lite 并连接
    #         for sfx in set_suffixes:
    #             inst = f"{node_name}_wstrm_{sfx}"
    #             axil = f"s_axilite_{sfx}"
    #             cmd.append(
    #                 f"if {{![llength [get_bd_intf_pins -quiet {parent}/{axil}]]}} "
    #                 f"{{ create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 {parent}/{axil} }}"
    #             )
    #             cmd.append(
    #                 f"connect_bd_intf_net [get_bd_intf_pins {parent}/{axil}] "
    #                 f"[get_bd_intf_pins {parent}/{inst}/s_axilite]"
    #             )

    #         cmd.append("assign_bd_address")
    #         cmd.append("save_bd_design")
    #         return cmd

    #     # =============== 动态权重（保持原逻辑） ===============
    #     if dyn_input:
    #         file_suffix = "_dynamic_load_wrapper.v"
    #         dynld_tmpl = None
    #         for fname in os.listdir(code_gen_dir):
    #             if fname.endswith(file_suffix):
    #                 dynld_tmpl = fname
    #                 break
    #         if dynld_tmpl is None:
    #             raise Exception(f"[DYNLOAD-IPI] Missing *{file_suffix} under {code_gen_dir}")

    #         dynld_tmpl_name = dynld_tmpl[:-2]
    #         sourcefiles = [
    #             os.path.join(code_gen_dir, dynld_tmpl),
    #             os.path.join(dynld_rtllib_dir, "ram_p_c.sv"),
    #             os.path.join(dynld_rtllib_dir, "dynamic_load.sv"),
    #         ]
    #         for f in sourcefiles:
    #             cmd += [f"add_files -copy_to {source_target} -force -norecurse {f}"]
    #         cmd.append("update_compile_order -fileset sources_1")

    #         dynld_inst = f"{node_name}_wdynld"
    #         cmd.append(f"create_bd_cell -type hier -reference {dynld_tmpl_name} {parent}/{dynld_inst}")

    #         # clocks & reset
    #         cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{dynld_inst}/ap_clk]")
    #         cmd.append(f"connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {parent}/{dynld_inst}/ap_rst_n]")

    #         # 额外暴露动态权重入口（第二路 s_axis）
    #         if len(vnames["s_axis"]) < 2:
    #             raise Exception("[DYNLOAD-IPI] HLS top doesn't expose secondary s_axis for dynamic load.")
    #         win_name = vnames["s_axis"][1][0]  # e.g., in1_V
    #         cmd.append(
    #             f"if {{![llength [get_bd_intf_pins -quiet {parent}/{win_name}]]}} "
    #             f"{{ create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 {parent}/{win_name} }}"
    #         )

    #         # AXIS：dynloader -> HLS in1_V
    #         cmd.append(
    #             f"connect_bd_intf_net [get_bd_intf_pins {parent}/{dynld_inst}/m_axis_0] "
    #             f"[get_bd_intf_pins {hlsop}/in1_V]"
    #         )
    #         # 上游 -> dynloader
    #         cmd.append(
    #             f"connect_bd_intf_net [get_bd_intf_pins {parent}/{win_name}] "
    #             f"[get_bd_intf_pins {parent}/{dynld_inst}/s_axis_0]"
    #         )

    #         cmd.append("save_bd_design")
    #         return cmd

    #     # =============== 稠密（单流 -> in1_V） ===============
    #     if mem_mode in ["internal_decoupled", "external"]:
    #         runtime_writable = (self.get_nodeattr("runtime_writeable_weights") == 1)

    #         # 找到唯一 *_memstream_wrapper.v
    #         candidates = [f for f in os.listdir(code_gen_dir) if f.endswith("_memstream_wrapper.v")]
    #         if len(candidates) != 1:
    #             raise Exception(
    #                 f"[DENSE-IPI] Expect exactly one '*_memstream_wrapper.v' under {code_gen_dir}, got {len(candidates)}"
    #             )
    #         strm_tmpl  = candidates[0]
    #         strm_modref= strm_tmpl[:-2]
    #         strm_inst  = f"{node_name}_wstrm"

    #         # add_files（-force）
    #         for f in [
    #             os.path.join(code_gen_dir, strm_tmpl),
    #             os.path.join(axi_dir, "axilite.sv"),
    #             os.path.join(ms_rtllib_dir, "memstream_axi.sv"),
    #             os.path.join(ms_rtllib_dir, "memstream.sv"),
    #         ]:
    #             cmd += [f"add_files -copy_to {source_target} -force -norecurse {f}"]
    #         cmd.append("update_compile_order -fileset sources_1")

    #         # 实例化 wrapper
    #         cmd.append(f"create_bd_cell -type hier -reference {strm_modref} {parent}/{strm_inst}")

    #         # clocks & reset
    #         cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{strm_inst}/ap_clk]")
    #         cmd.append(f"connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {parent}/{strm_inst}/ap_rst_n]")
    #         # ap_clk2x：按 dense.tcl 的做法，缺省绑到 ap_clk；如确需 2x，再开启 pumped_mem
    #         if pumped_mem:
    #             cmd.append(f"connect_bd_net [get_bd_pins {parent}/ap_clk2x] [get_bd_pins {parent}/{strm_inst}/ap_clk2x]")
    #         else:
    #             cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{strm_inst}/ap_clk2x]")

    #         # AXIS：wrapper -> HLS in1_V
    #         cmd.append(
    #             f"connect_bd_intf_net [get_bd_intf_pins {parent}/{strm_inst}/m_axis_0] "
    #             f"[get_bd_intf_pins {hlsop}/in1_V]"
    #         )

    #         # 仅 runtime 可写时暴露 AXI-Lite
    #         if runtime_writable:
    #             axilite_name = "s_axilite"
    #             cmd.append(
    #                 f"if {{![llength [get_bd_intf_pins -quiet {parent}/{axilite_name}]]}} "
    #                 f"{{ create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 {parent}/{axilite_name} }}"
    #             )
    #             cmd.append(
    #                 f"connect_bd_intf_net [get_bd_intf_pins {parent}/{axilite_name}] "
    #                 f"[get_bd_intf_pins {parent}/{strm_inst}/{axilite_name}]"
    #             )
    #             cmd.append("assign_bd_address")

    #         cmd.append("save_bd_design")
    #         return cmd

    #     # 其它 mem_mode：只保存
    #     cmd.append("save_bd_design")
    #     return cmd
    def code_generation_ipi(self):
        try:
            sparse_mode = self.get_nodeattr("sparse_mode")
        except AttributeError:
            sparse_mode = "dense"

        if sparse_mode == "spmv_sparse":
            cmd = self.code_generation_ipi_hls()
        else:
            cmd = self.code_generation_ipi_default()

        return cmd

    def code_generation_ipi_default(self):
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        dyn_input = self.get_nodeattr("dynamic_input")
        mem_mode = self.get_nodeattr("mem_mode")
        sname = "V"

        # check if additional components are needed
        if dyn_input or mem_mode == "internal_decoupled":
            node_name = self.onnx_node.name
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            # clock and reset
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            # if we need a 2x clock for either compute or memory, instantiate the 2x clk port
            try:
                pumped_compute = self.get_nodeattr("pumpedCompute")
            except AttributeError:
                pumped_compute = 0
            if pumped_compute or self.get_nodeattr("pumpedMemory"):
                clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
                cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk2x_name))
            else:
                clk2x_name = None
            # streams
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )
            # instantiate the RTL block
            # Instantiate either the HLS or RTL IP depending on operator
            self.instantiate_ip(cmd)
            # connect MVAU
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, rst_name, node_name, node_name, rst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, clk_name, node_name, node_name, clk_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, din_name, node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, dout_name, node_name, node_name, dout_name)
            )

            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            if dyn_input:
                # dynamic loader
                dynld_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/dynload/hdl/")
                file_suffix = "_dynamic_load_wrapper.v"
                # automatically find memstream verilog component in code generation directory
                for fname in os.listdir(code_gen_dir):
                    if fname.endswith(file_suffix):
                        dynld_tmpl = fname
                dynld_tmpl_name = dynld_tmpl[:-2]
                sourcefiles = [
                    os.path.join(code_gen_dir, dynld_tmpl),
                    dynld_rtllib_dir + "ram_p_c.sv",
                    dynld_rtllib_dir + "dynamic_load.sv",
                ]
                for f in sourcefiles:
                    cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
                dynld_inst = node_name + "_wdynld"
                # instantiate the cell
                cmd.append(
                    "create_bd_cell -type hier -reference %s /%s/%s"
                    % (dynld_tmpl_name, node_name, dynld_inst)
                )
                # additional dynamic input
                win_name = self.get_verilog_top_module_intf_names()["s_axis"][1][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, win_name)
                )
                # connect
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                    % (node_name, clk_name, node_name, dynld_inst)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                    % (node_name, rst_name, node_name, dynld_inst)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                    "[get_bd_intf_pins %s/%s/in1_%s]"
                    % (node_name, dynld_inst, node_name, node_name, sname)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/s_axis_0]"
                    % (node_name, win_name, node_name, dynld_inst)
                )
            else:
                # memstream
                runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
                axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
                ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
                file_suffix = "_memstream_wrapper.v"
                # automatically find memstream verilog component in code generation directory
                for fname in os.listdir(code_gen_dir):
                    if fname.endswith(file_suffix):
                        strm_tmpl = fname
                strm_tmpl_name = strm_tmpl[:-2]
                sourcefiles = [
                    os.path.join(code_gen_dir, strm_tmpl),
                    axi_dir + "axilite.sv",
                    ms_rtllib_dir + "memstream_axi.sv",
                    ms_rtllib_dir + "memstream.sv",
                ]
                for f in sourcefiles:
                    cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
                strm_inst = node_name + "_wstrm"
                # instantiate the cell
                cmd.append(
                    "create_bd_cell -type hier -reference %s /%s/%s"
                    % (strm_tmpl_name, node_name, strm_inst)
                )
                # connect
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                    % (node_name, clk_name, node_name, strm_inst)
                )
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                    % (node_name, rst_name, node_name, strm_inst)
                )
                # if using 2x pumped memory, connect the memstreamer's 2x clk input
                # to the 2x clock port. otherwise connect it to the regular clock port.
                if self.get_nodeattr("pumpedMemory"):
                    cmd.append(
                        "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                        % (node_name, clk2x_name, node_name, strm_inst)
                    )
                else:
                    cmd.append(
                        "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                        % (node_name, clk_name, node_name, strm_inst)
                    )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                    "[get_bd_intf_pins %s/%s/in1_%s]"
                    % (node_name, strm_inst, node_name, node_name, sname)
                )
                # runtime writeable weights
                if runtime_writable:
                    axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                    cmd.append(
                        "create_bd_intf_pin -mode Slave "
                        "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s"
                        % (node_name, axilite_name)
                    )
                    cmd.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                        "[get_bd_intf_pins %s/%s/%s]"
                        % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                    )
                    # TODO calculate and pass in segment size here
                    cmd.append("assign_bd_address")

            # save bd
            cmd.append("save_bd_design")
        elif mem_mode == "internal_embedded" or mem_mode == "external":
            # base class impl sufficient for internal_embedded/external modes
            self.instantiate_ip(cmd)
        else:
            raise Exception("Unrecognized mem_mode for MatrixVectorActivation")
        return cmd
    
    def code_generation_ipi_hls(self):
        """
        生成本层 IPI 连接（Vivado Tcl 命令列表）。
        - 先确保 /<node> 父层级与 /<node>/<node> HLS IP 存在，并完成基础连线
        - 稀疏(sparse_mode=="spmv_sparse")：4 路 memstream -> <sfx>_V （仅新命名）
        - 稠密(默认)：单 memstream -> in1_V
        - 动态权重(dynamic_input)：使用 *_dynamic_load_wrapper.v
        说明：
        稀疏 wrapper 仅兼容新命名：<node>_{sfidx|val|mask|rowlen}_memstream_wrapper.v（模块名=文件名去 .v）
        """
        import os

        cmd = []

        FINN_ROOT = os.environ.get("FINN_ROOT", "")
        node_name     = self.onnx_node.name
        parent        = f"/{node_name}"
        hlsop         = f"/{node_name}/{node_name}"
        source_target = f"./ip/verilog/rtl_ops/{node_name}"
        code_gen_dir  = self.get_nodeattr("code_gen_dir_ipgen")

        # rtllib 路径
        axi_dir          = os.path.join(FINN_ROOT, "finn-rtllib/axi/hdl/")
        ms_rtllib_dir    = os.path.join(FINN_ROOT, "finn-rtllib/memstream/hdl/")
        dynld_rtllib_dir = os.path.join(FINN_ROOT, "finn-rtllib/dynload/hdl/")

        # 属性
        dyn_input   = bool(self.get_nodeattr("dynamic_input"))
        mem_mode    = self.get_nodeattr("mem_mode")
        pumped_mem  = bool(self.get_nodeattr("pumpedMemory"))
        try:
            sparse_mode = self.get_nodeattr("sparse_mode")
        except Exception:
            sparse_mode = "dense"

        # 端口名（从 HLS 顶层抽取，避免硬编码）
        vnames      = self.get_verilog_top_module_intf_names()
        clk_name    = vnames["clk"][0]                 # 通常 "ap_clk"
        rst_name    = vnames["rst"][0]                 # 通常 "ap_rst_n"
        din_name    = vnames["s_axis"][0][0]           # 通常 "in0_V"
        dout_name   = vnames["m_axis"][0][0]           # 通常 "out0_V"

        # ---- 打开/创建 BD + 父层级 + HLS IP，并完成基础连线 ----
        cmd += [
            # 确保 BD 打开
            'if {[catch {current_bd_design} _cur_bd]} { set _cur_bd "" }',
            'if {$_cur_bd eq ""} { create_bd_design "finn_design" }',

            # 目标目录
            f'file mkdir {source_target}',

            # 父层级 /<node>（若无则创建）与其顶层 pin/接口（幂等创建）
            f'if {{![llength [get_bd_cells -quiet {parent}]]}} {{ create_bd_cell -type hier {node_name} }}',
            f'if {{![llength [get_bd_pins -quiet {parent}/{clk_name}]]}} {{ create_bd_pin -dir I -type clk {parent}/{clk_name} }}',
            f'if {{![llength [get_bd_pins -quiet {parent}/{rst_name}]]}} {{ create_bd_pin -dir I -type rst {parent}/{rst_name} }}',
            # ap_clk2x：仅在 pumped_mem 时暴露
            (f'if {{![llength [get_bd_pins -quiet {parent}/ap_clk2x]]}} {{ create_bd_pin -dir I -type clk {parent}/ap_clk2x }}'
            if pumped_mem else '# no ap_clk2x on parent'),

            f'if {{![llength [get_bd_intf_pins -quiet {parent}/{dout_name}]]}} {{ '
            f'  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 {parent}/{dout_name} }}',
            f'if {{![llength [get_bd_intf_pins -quiet {parent}/{din_name}]]}} {{ '
            f'  create_bd_intf_pin -mode Slave  -vlnv xilinx.com:interface:axis_rtl:1.0 {parent}/{din_name} }}',

            # HLS IP /<node>/<node>（若无则从 IP Catalog 实例化 xilinx.com:hls:<node>:*）
            f'if {{![llength [get_bd_cells -quiet {hlsop}]]}} {{ '
            f'  set _defs [get_ipdefs -all -filter "VLNV =~ xilinx.com:hls:{node_name}:*"]; '
            f'  if {{[llength $_defs]}} {{ create_bd_cell -type ip -vlnv [lindex $_defs 0] {hlsop} }} '
            f'  else {{ error {{HLS IP xilinx.com:hls:{node_name}:* not found}} }} '
            f'}}',

            # 连接父层级 与 HLS 顶层端口/接口
            f'if {{[llength [get_bd_pins -quiet {hlsop}/{rst_name}]]}} '
            f'{{ connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {hlsop}/{rst_name}] }}',
            f'if {{[llength [get_bd_pins -quiet {hlsop}/{clk_name}]]}} '
            f'{{ connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {hlsop}/{clk_name}] }}',
            f'if {{[llength [get_bd_intf_pins -quiet {hlsop}/{din_name}]]}} '
            f'{{ connect_bd_intf_net [get_bd_intf_pins {parent}/{din_name}] [get_bd_intf_pins {hlsop}/{din_name}] }}',
            f'if {{[llength [get_bd_intf_pins -quiet {hlsop}/{dout_name}]]}} '
            f'{{ connect_bd_intf_net [get_bd_intf_pins {parent}/{dout_name}] [get_bd_intf_pins {hlsop}/{dout_name}] }}',
        ]

        # =============== 稀疏（仅新命名） ===============
        if (sparse_mode == "spmv_sparse") and (not dyn_input):
            set_suffixes = ["sfidx", "val", "mask", "rowlen"]

            # 校验 4 个 wrapper 文件存在；模块名=文件名去 .v
            for sfx in set_suffixes:
                vfile = f"{node_name}_{sfx}_memstream_wrapper.v"
                vpath = os.path.join(code_gen_dir, vfile)
                if not os.path.exists(vpath):
                    raise Exception(
                        f"[SPMV-IPI] Missing {vpath}. Only NEW naming supported. "
                        f"Please run generate_hdl_memstream_spmv() first."
                    )

            for sfx in set_suffixes:
                vfile  = f"{node_name}_{sfx}_memstream_wrapper.v"
                modref = vfile[:-2]                   # module == filename without ".v"
                inst   = f"{node_name}_wstrm_{sfx}"

                # add_files（带 -force）+ 刷新编译顺序
                for f in [
                    os.path.join(code_gen_dir, vfile),
                    os.path.join(axi_dir, "axilite.sv"),
                    os.path.join(ms_rtllib_dir, "memstream_axi.sv"),
                    os.path.join(ms_rtllib_dir, "memstream.sv"),
                ]:
                    cmd += [f"add_files -copy_to {source_target} -force -norecurse {f}"]
                cmd.append("update_compile_order -fileset sources_1")

                # 实例化 wrapper
                cmd.append(f"create_bd_cell -type hier -reference {modref} {parent}/{inst}")

                # clocks & reset
                cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{inst}/ap_clk]")
                cmd.append(f"connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {parent}/{inst}/ap_rst_n]")
                if pumped_mem:
                    cmd.append(f"connect_bd_net [get_bd_pins {parent}/ap_clk2x] [get_bd_pins {parent}/{inst}/ap_clk2x]")
                else:
                    cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{inst}/ap_clk2x]")

                # AXIS：wrapper -> HLS <sfx>_V
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {parent}/{inst}/m_axis_0] "
                    f"[get_bd_intf_pins {hlsop}/{sfx}_V]"
                )

            # 暴露 4 个 AXI-Lite 并连接
            for sfx in set_suffixes:
                inst = f"{node_name}_wstrm_{sfx}"
                axil = f"s_axilite_{sfx}"
                cmd.append(
                    f"if {{![llength [get_bd_intf_pins -quiet {parent}/{axil}]]}} "
                    f"{{ create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 {parent}/{axil} }}"
                )
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {parent}/{axil}] "
                    f"[get_bd_intf_pins {parent}/{inst}/s_axilite]"
                )

            cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
            return cmd

        # =============== 动态权重（保持原逻辑） ===============
        if dyn_input:
            file_suffix = "_dynamic_load_wrapper.v"
            dynld_tmpl = None
            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    dynld_tmpl = fname
                    break
            if dynld_tmpl is None:
                raise Exception(f"[DYNLOAD-IPI] Missing *{file_suffix} under {code_gen_dir}")

            dynld_tmpl_name = dynld_tmpl[:-2]
            sourcefiles = [
                os.path.join(code_gen_dir, dynld_tmpl),
                os.path.join(dynld_rtllib_dir, "ram_p_c.sv"),
                os.path.join(dynld_rtllib_dir, "dynamic_load.sv"),
            ]
            for f in sourcefiles:
                cmd += [f"add_files -copy_to {source_target} -force -norecurse {f}"]
            cmd.append("update_compile_order -fileset sources_1")

            dynld_inst = f"{node_name}_wdynld"
            cmd.append(f"create_bd_cell -type hier -reference {dynld_tmpl_name} {parent}/{dynld_inst}")

            # clocks & reset
            cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{dynld_inst}/ap_clk]")
            cmd.append(f"connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {parent}/{dynld_inst}/ap_rst_n]")

            # 额外暴露动态权重入口（第二路 s_axis）
            if len(vnames["s_axis"]) < 2:
                raise Exception("[DYNLOAD-IPI] HLS top doesn't expose secondary s_axis for dynamic load.")
            win_name = vnames["s_axis"][1][0]  # e.g., in1_V
            cmd.append(
                f"if {{![llength [get_bd_intf_pins -quiet {parent}/{win_name}]]}} "
                f"{{ create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 {parent}/{win_name} }}"
            )

            # AXIS：dynloader -> HLS in1_V
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {parent}/{dynld_inst}/m_axis_0] "
                f"[get_bd_intf_pins {hlsop}/in1_V]"
            )
            # 上游 -> dynloader
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {parent}/{win_name}] "
                f"[get_bd_intf_pins {parent}/{dynld_inst}/s_axis_0]"
            )

            cmd.append("save_bd_design")
            return cmd

        # =============== 稠密（单流 -> in1_V） ===============
        if mem_mode in ["internal_decoupled", "external"]:
            runtime_writable = (self.get_nodeattr("runtime_writeable_weights") == 1)

            # 找到唯一 *_memstream_wrapper.v
            candidates = [f for f in os.listdir(code_gen_dir) if f.endswith("_memstream_wrapper.v")]
            if len(candidates) != 1:
                raise Exception(
                    f"[DENSE-IPI] Expect exactly one '*_memstream_wrapper.v' under {code_gen_dir}, got {len(candidates)}"
                )
            strm_tmpl  = candidates[0]
            strm_modref= strm_tmpl[:-2]
            strm_inst  = f"{node_name}_wstrm"

            # add_files（-force）
            for f in [
                os.path.join(code_gen_dir, strm_tmpl),
                os.path.join(axi_dir, "axilite.sv"),
                os.path.join(ms_rtllib_dir, "memstream_axi.sv"),
                os.path.join(ms_rtllib_dir, "memstream.sv"),
            ]:
                cmd += [f"add_files -copy_to {source_target} -force -norecurse {f}"]
            cmd.append("update_compile_order -fileset sources_1")

            # 实例化 wrapper
            cmd.append(f"create_bd_cell -type hier -reference {strm_modref} {parent}/{strm_inst}")

            # clocks & reset
            cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{strm_inst}/ap_clk]")
            cmd.append(f"connect_bd_net [get_bd_pins {parent}/{rst_name}] [get_bd_pins {parent}/{strm_inst}/ap_rst_n]")
            # ap_clk2x：按 dense.tcl 的做法，缺省绑到 ap_clk；如确需 2x，再开启 pumped_mem
            if pumped_mem:
                cmd.append(f"connect_bd_net [get_bd_pins {parent}/ap_clk2x] [get_bd_pins {parent}/{strm_inst}/ap_clk2x]")
            else:
                cmd.append(f"connect_bd_net [get_bd_pins {parent}/{clk_name}] [get_bd_pins {parent}/{strm_inst}/ap_clk2x]")

            # AXIS：wrapper -> HLS in1_V
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {parent}/{strm_inst}/m_axis_0] "
                f"[get_bd_intf_pins {hlsop}/in1_V]"
            )

            # 仅 runtime 可写时暴露 AXI-Lite
            if runtime_writable:
                axilite_name = "s_axilite"
                cmd.append(
                    f"if {{![llength [get_bd_intf_pins -quiet {parent}/{axilite_name}]]}} "
                    f"{{ create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 {parent}/{axilite_name} }}"
                )
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {parent}/{axilite_name}] "
                    f"[get_bd_intf_pins {parent}/{strm_inst}/{axilite_name}]"
                )
                cmd.append("assign_bd_address")

            cmd.append("save_bd_design")
            return cmd

        # 其它 mem_mode：只保存
        cmd.append("save_bd_design")
        return cmd
