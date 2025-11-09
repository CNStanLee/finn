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
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.util.basic import is_versal
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

# ONNX i/o tensor shape assumptions for MatrixVectorActivation_hls:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)


class MVAU_hls(MVAU, HLSBackend):
    """Corresponds to finn-hlslib MatrixVectorActivation_Batch function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(MVAU.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        # for HLS MVAU default resType to lut
        my_attrs["resType"] = ("s", False, "lut", {"auto", "lut", "dsp"})
        return my_attrs

    def lut_estimation(self):
        """Calculates resource estimations for LUTs based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018
        """
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        MW = self.get_nodeattr("MW")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        # determine tdt with input and weight data types
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        # parameters from experiments in paper mentioned above
        c0 = 300
        c1 = 1.1
        c2 = 0
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")
        if (mmode == "internal_decoupled" and mstyle == "distributed") or (
            mmode == "internal_embedded" and self.calc_wmem() <= 128
        ):
            c2 = (P * Q * W) * math.ceil(self.calc_wmem() / 64)

        # multiplication
        res_type = self.get_nodeattr("resType")
        if res_type == "dsp":
            mult_luts = 0
        else:
            mult_luts = Q * (2 * math.ceil((W + A) / 6) - 1) * (W + A)
        # adder tree
        addertree_luts = (W + A) * (2 * Q - 1)
        # accumulator
        acc_datatype = self.get_accumulator_datatype()
        # if accDataType is not set, then it will default to INT32, which would
        # be a large overestimate in most (if not all) cases. In this scenario,
        # we would use the minimum accumulator as determined by the data types
        # bound, derived in https://arxiv.org/abs/2301.13376
        alpha = math.log(MW, 2) + W + A - 1 - int(idt.signed())
        acc_bits = min(
            acc_datatype.bitwidth(),
            np.ceil(alpha + math.log(1 + pow(2, -alpha), 2) + 1),
        )
        acc_luts = acc_bits
        # thresholds and threshold comparators
        thr_luts = 0
        comp_luts = 0
        noact = self.get_nodeattr("noActivation")
        tmem_style = self.get_nodeattr("ram_style_thresholds")
        if (noact == 0) and (tmem_style == "distributed"):
            odt = self.get_output_datatype()
            B = odt.bitwidth()
            thr_luts = (2**B - 1) * acc_bits * math.ceil(self.calc_tmem() / 64)
            comp_luts = (2**B - 1) * acc_bits

        return int(
            c0 + c1 * (P * (mult_luts + addertree_luts + acc_luts + thr_luts + comp_luts)) + c2
        )

    def dsp_estimation(self, fpgapart):
        # multiplication
        P = self.get_nodeattr("PE")
        res_type = self.get_nodeattr("resType")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        if res_type == "dsp":
            mult_dsp = P * Q * np.ceil((W + A) / 48)  # TODO: more accurate modelling
        else:
            mult_dsp = 0
        return int(mult_dsp)

    # matrixvectoractivation_hls.py 里，替换整个 code_generation_ipgen
    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates c++ code and tcl script for ip generation."""
        super().code_generation_ipgen(model, fpgapart, clk)
        dynamic_input = self.get_nodeattr("dynamic_input")
        mem_mode = self.get_nodeattr("mem_mode")

        if dynamic_input:
            self.generate_hdl_dynload()

        if mem_mode == "internal_decoupled":
            if self.get_nodeattr("ram_style") == "ultra" and not is_versal(fpgapart):
                runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
                assert (
                    runtime_writeable == 1
                ), """Layer with URAM weights must have runtime_writeable_weights=1
                    if Ultrascale device is targeted."""

            # --- 新增：spmv_sparse 时生成 4 个 wrapper ---
            try:
                sparse_mode = self.get_nodeattr("sparse_mode")
            except AttributeError:
                sparse_mode = "dense"

            if sparse_mode == "spmv_sparse":
                # 4 路 memstream wrapper
                self.generate_hdl_memstream_spmv(
                    fpgapart, pumped_memory=self.get_nodeattr("pumpedMemory")
                )
            else:
                # 原稠密路径：单个 memstream wrapper
                self.generate_hdl_memstream(
                    fpgapart, pumped_memory=self.get_nodeattr("pumpedMemory")
                )

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        # out_is_binary = self.get_output_datatype() == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.get_nodeattr("binaryXnorMode") == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise Exception("True binary (non-bipolar) inputs not yet supported")
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        # out_is_bipolar = self.get_output_datatype() == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # fill in TSrcI and TWeightI
        # TODO check these with Giulio
        # TODO handle non-bipolar binary inputs
        if inp_is_bipolar and wt_is_bipolar:
            ret["TSrcI"] = "Recast<XnorMul>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and wt_is_bipolar:
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Recast<Binary>"
        elif inp_is_bipolar and (not wt_is_bipolar):
            ret["TSrcI"] = "Recast<Binary>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and (not wt_is_bipolar):
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Identity"

        # fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode not in ["internal_embedded", "internal_decoupled", "external"]:
            raise Exception(
                """Please set mem_mode to "internal_embedded", "internal_decoupled", or "external",
                currently no other parameter value is supported!"""
            )
        self.code_gen_dict["$GLOBALS$"] += ['#include "mvau.hpp"']
        # if spare mode is set to 'spmv_sparse', include the sparse header
        try:
            sparse_mode = self.get_nodeattr("sparse_mode")
        except AttributeError:
            sparse_mode = "dense"
        if sparse_mode == "spmv_sparse":
            self.code_gen_dict["$GLOBALS$"] += ['#include "sfcsr_mvau.hpp"']

        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var):
        # Only ipgen mode: Make sure that SIMD parameter satisfies minimum requirements.
        if var == "ipgen":
            SIMD = self.get_nodeattr("SIMD")
            MW = self.get_nodeattr("MW")
            condition = SIMD >= (MW / 1024)
            msg = (
                f"HLS synthesis of MatrixVectorActivation requires: "
                f"SIMD >= MW / 1024. This is not fulfilled with: SIMD={SIMD} "
                f"and MW={MW} for node: {self.onnx_node.name}."
            )
            assert condition, msg
        mem_mode = self.get_nodeattr("mem_mode")
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = np.prod(numInputVectors)
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW1 {}\n #define MH1 {}\n
            #define SIMD1 {}\n #define PE1 {}\n #define WMEM1 {}\n
            #define TMEM1 {}\n #define numReps {}""".format(
                self.get_nodeattr("MW"),
                self.get_nodeattr("MH"),
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("PE"),
                self.calc_wmem(),
                self.calc_tmem(),
                numReps,
            )
        ]
        if mem_mode == "internal_decoupled" or mem_mode == "external":
            wdt = self.get_input_datatype(1)
            self.code_gen_dict["$DEFINES$"].append("#define WP1 {}\n".format(wdt.bitwidth()))

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

        # 输入数据 in0_V
        dtype = self.get_input_datatype(0)
        if dtype == DataType["BIPOLAR"]:
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width(0)
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        # 注意输入的 innermost 维反向
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0_V, false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

        # 权值 / 稀疏四流
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled" or mem_mode == "external":
            # 判断是否稀疏
            try:
                sparse_mode = self.get_nodeattr("sparse_mode")
            except AttributeError:
                sparse_mode = "dense"
            is_spmv = (sparse_mode == "spmv_sparse")

            if is_spmv:
                MW   = int(self.get_nodeattr("MW"))
                SIMD = int(self.get_nodeattr("SIMD"))
                PE   = int(self.get_nodeattr("PE"))
                ncols_per_lane = max(1, (MW + SIMD - 1) // SIMD)
                sfidx_width = max(1, math.ceil(math.log2(ncols_per_lane)))

                # 权值位宽（BIPOLAR 导出为 BINARY）
                wdt = self.get_input_datatype(1)
                export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
                wbits = export_wdt.bitwidth()
                w_elem_hls = export_wdt.get_hls_datatype_str()

                # sfidx
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2apintstream<ap_uint<%d>, ap_uint<%d>, %d, %s>("%s/input_sfidx.npy", sfidx_V, false, numReps);'
                    % (PE*SIMD*sfidx_width, sfidx_width, sfidx_width, "float", code_gen_dir)
                )
                # val
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2apintstream<ap_uint<%d>, %s, %d, %s>("%s/input_val.npy", val_V, false, numReps);'
                    % (PE*SIMD*wbits, w_elem_hls, wbits, "float", code_gen_dir)
                )
                # mask（1bit）
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2apintstream<ap_uint<%d>, ap_uint<1>, %d, %s>("%s/input_mask.npy", mask_V, false, numReps);'
                    % (PE*SIMD, 1, "float", code_gen_dir)
                )
                # rowlen（每 PE 16bit）
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2apintstream<ap_uint<%d>, ap_uint<16>, %d, %s>("%s/input_rowlen.npy", rowlen_V, false, numReps);'
                    % (PE*16, 16, "float", code_gen_dir)
                )
            else:
                # 稠密：读 input_1.npy → in1_V
                wdt = self.get_input_datatype(1)
                elem_bits = wdt.bitwidth()
                packed_bits = self.get_instream_width(1)
                if self.get_nodeattr("dynamic_input"):
                    packed_bits = packed_bits * self.get_nodeattr("SIMD")
                packed_hls_type = "ap_uint<%d>" % packed_bits
                elem_hls_type = wdt.get_hls_datatype_str()
                npy_in = "%s/input_1.npy" % code_gen_dir
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2apintstream<%s, %s, %d, %s>("%s", in1_V, false, numReps);'
                    % (packed_hls_type, elem_hls_type, elem_bits, "float", npy_in)
                )

    def strm_decl(self):
        mem_mode = self.get_nodeattr("mem_mode")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        # 固定的数据输入与输出
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width(0))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )

        if mem_mode == "internal_decoupled" or mem_mode == "external":
            # 判断是否稀疏
            try:
                sparse_mode = self.get_nodeattr("sparse_mode")
            except AttributeError:
                sparse_mode = "dense"
            is_spmv = (sparse_mode == "spmv_sparse")

            if is_spmv:
                # 计算每个 lane 的列块数 -> sf 索引位宽
                MW   = int(self.get_nodeattr("MW"))
                SIMD = int(self.get_nodeattr("SIMD"))
                PE   = int(self.get_nodeattr("PE"))
                ncols_per_lane = max(1, (MW + SIMD - 1) // SIMD)
                sfidx_width = max(1, math.ceil(math.log2(ncols_per_lane)))

                # 权值位宽（BIPOLAR 导出为 BINARY=1bit）
                wdt = self.get_input_datatype(1)
                export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
                wbits = export_wdt.bitwidth()

                # 四路流声明
                self.code_gen_dict["$STREAMDECLARATIONS$"] += [
                    f'hls::stream<ap_uint<{PE}*{SIMD}*{sfidx_width}>> sfidx_V ("sfidx_V");',
                    f'hls::stream<ap_uint<{PE}*{SIMD}*{wbits}>> val_V ("val_V");',
                    f'hls::stream<ap_uint<{PE}*{SIMD}>> mask_V ("mask_V");',
                    f'hls::stream<ap_uint<{PE}*16>> rowlen_V ("rowlen_V");',
                ]
            else:
                # 稠密：仍然只声明 in1_V
                iwidth = self.get_instream_width(1)
                if self.get_nodeattr("dynamic_input"):
                    iwidth = iwidth * self.get_nodeattr("SIMD")
                self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                    'hls::stream<ap_uint<{}>> in1_V ("in1_V");'.format(iwidth)
                )

    def docompute(self):
        import math

        mem_mode = self.get_nodeattr("mem_mode")
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }
        tmpl_args = self.get_template_param_values()

        # 激活对象保持原逻辑
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            activation_expr = "PassThroughActivation<%s>()" % odtype_hls_str
            ta_tmpl = "PassThroughActivation<%s>" % odtype_hls_str
        else:
            activation_expr = "threshs"
            ta_tmpl = "decltype(threshs)"

        if mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1, {TSrcI}, {TDstI}, {TWeightI}>
    (in0_V, out0_V, weights, {threshs}, numReps, {resType});""".format(
                    TSrcI=tmpl_args["TSrcI"],
                    TDstI=tmpl_args["TDstI"],
                    TWeightI=tmpl_args["TWeightI"],
                    threshs=activation_expr,
                    resType=map_to_hls_mult_style[self.get_nodeattr("resType")],
                )
            ]
            return

        elif mem_mode in ["internal_decoupled", "external"]:
            # 稠密流式备用调用
            wdt = self.get_input_datatype(1)
            export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
            wdtype_hls_str = export_wdt.get_hls_datatype_str()

            dense_call = """Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, {TSrcI}, {TDstI}, {TWeightI}, {WDT}>
    (in0_V, out0_V, in1_V, {threshs}, numReps, {resType});""".format(
                TSrcI=tmpl_args["TSrcI"],
                TDstI=tmpl_args["TDstI"],
                TWeightI=tmpl_args["TWeightI"],
                WDT=wdtype_hls_str,
                threshs=activation_expr,
                resType=map_to_hls_mult_style[self.get_nodeattr("resType")],
            )

            # 稀疏开关
            try:
                sparse_mode = self.get_nodeattr("sparse_mode")
            except AttributeError:
                sparse_mode = "dense"
            is_spmv = (sparse_mode == "spmv_sparse")

            if not is_spmv:
                self.code_gen_dict["$DOCOMPUTE$"] = [dense_call]
                return

            # ====== SPMV 分支：显式模板参数 + out 最后 ======
            MW   = int(self.get_nodeattr("MW"))
            SIMD = int(self.get_nodeattr("SIMD"))

            # Sf 索引位宽 = ceil(log2(ceil(MW/SIMD)))（至少 1）
            ncols_per_lane = max(1, (MW + SIMD - 1) // SIMD)
            SfIdxWidth = max(1, math.ceil(math.log2(ncols_per_lane)))

            # 输入/输出/权值位宽（BIPOLAR->BINARY）
            in_dt0 = self.get_input_datatype(0)
            in_dt  = DataType["BINARY"] if in_dt0 == DataType["BIPOLAR"] else in_dt0
            INPUT_PRECISION = in_dt.bitwidth()

            out_dt = self.get_output_datatype()
            ACTIVATION_PRECISION = out_dt.bitwidth()

            WBITS = export_wdt.bitwidth()

            # 需要的 include（保持本地，不动 globals）
            self.code_gen_dict.setdefault("$INCLUDES$", [])
            if '#include "sfcsr_mvau.hpp"' not in self.code_gen_dict["$INCLUDES$"]:
                self.code_gen_dict["$INCLUDES$"].append('#include "sfcsr_mvau.hpp"')

            # 模板参数显式给全；TO=csr_pe_act_t（由 blackboxfunction 注入的本地类型）
            spmv_call = """sfcsr_mvau<
MW1, MH1, SIMD1, PE1, {SfW},
{TSrcI}, {TDstI}, {TWeightI},
ap_uint<SIMD1*{INP}>,
csr_pe_act_t,
{TA},
ap_uint<{WB}>
>(in0_V, sfidx_V, val_V, mask_V, rowlen_V, {threshs}, numReps, {resType}, out0_V);""".format(
    SfW=SfIdxWidth,                 # ceil(log2(ceil(MW/SIMD)))
    TSrcI=tmpl_args["TSrcI"],       # 保持原模板的数据接口类型
    TDstI=tmpl_args["TDstI"],
    TWeightI=tmpl_args["TWeightI"],
    INP=INPUT_PRECISION,            # 输入元素位宽（BIPOLAR->1）
    TA=ta_tmpl,                     # PassThroughActivation<ODT> 或 decltype(threshs)
    WB=WBITS,                       # TW = ap_uint<WBITS>（权值元素位宽）
    threshs=activation_expr,        # 激活对象作为函数实参传入
    resType=map_to_hls_mult_style[self.get_nodeattr("resType")]  # ← 注意：这里必须带括号，例如 ap_resource_lut()
)
            self.code_gen_dict["$DOCOMPUTE$"] = [spmv_call]
            return

        else:
            raise Exception(
                """Please set mem_mode to "internal_embedded", "internal_decoupled", or "external",
    currently no other parameter value is supported!"""
            )







    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output_0.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out0_V, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                shape_cpp_str,
                npy_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        import math
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_V,
        hls::stream<ap_uint<{}>> &out0_V
    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(0),
                    self.get_outstream_width(),
                )
            ]
            return

        elif mem_mode in ["internal_decoupled", "external"]:
            # 稀疏？
            try:
                sparse_mode = self.get_nodeattr("sparse_mode")
            except AttributeError:
                sparse_mode = "dense"
            is_spmv = (sparse_mode == "spmv_sparse")

            if is_spmv:
                MW   = int(self.get_nodeattr("MW"))
                SIMD = int(self.get_nodeattr("SIMD"))
                PE   = int(self.get_nodeattr("PE"))

                ncols_per_lane = max(1, (MW + SIMD - 1) // SIMD)
                SfIdxWidth = max(1, math.ceil(math.log2(ncols_per_lane)))

                # 权值位宽（BIPOLAR -> BINARY）
                wdt = self.get_input_datatype(1)
                export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
                WBits = export_wdt.bitwidth()

                # 在签名前“就地”定义本地 csr_pe_act_t（非模板），位宽用数字常量，保证类型在函数签名可见
                act_bits = self.get_output_datatype().bitwidth()
                local_typedef = """
    // local csr_pe_act_t: non-template; width = PE1*%d; must be visible to signature
    struct csr_pe_act_t {
        ap_uint<PE1*%d> v;
        inline ap_range_ref<PE1*%d,false> operator()(unsigned pe, unsigned /*mmv*/, unsigned /*en*/) {
            return v.range((pe+1)*%d-1, pe*%d);
        }
        inline ap_range_ref<PE1*%d,false> operator()(unsigned pe, unsigned /*mmv*/, unsigned /*en*/) const {
            return const_cast<ap_uint<PE1*%d>&>(v).range((pe+1)*%d-1, pe*%d);
        }
        inline ap_range_ref<PE1*%d,false> operator()(unsigned pe, unsigned /*mmv*/) {
            return v.range((pe+1)*%d-1, pe*%d);
        }
        inline ap_range_ref<PE1*%d,false> operator()(unsigned pe, unsigned /*mmv*/) const {
            return const_cast<ap_uint<PE1*%d>&>(v).range((pe+1)*%d-1, pe*%d);
        }
        inline operator ap_uint<PE1*%d>() const { return v; }
    };
    """ % (act_bits, act_bits, act_bits, act_bits, act_bits,
        act_bits, act_bits, act_bits, act_bits,
        act_bits, act_bits, act_bits,
        act_bits, act_bits, act_bits, act_bits, act_bits)

                self.code_gen_dict["$BLACKBOXFUNCTION$"] = [local_typedef + """
    void {}(
        hls::stream<ap_uint<{}>> &in0_V,
        hls::stream<ap_uint<{}>> &sfidx_V,
        hls::stream<ap_uint<{}>> &val_V,
        hls::stream<ap_uint<{}>> &mask_V,
        hls::stream<ap_uint<{}>> &rowlen_V,
        hls::stream<csr_pe_act_t> &out0_V
    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(0),
                    PE*SIMD*SfIdxWidth,
                    PE*SIMD*WBits,
                    PE*SIMD,
                    PE*16,
                )]
                return

            else:
                wwidth = self.get_instream_width(1)
                if self.get_nodeattr("dynamic_input"):
                    wwidth = wwidth * self.get_nodeattr("SIMD")
                self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                    """void {}(
        hls::stream<ap_uint<{}>> &in0_V,
        hls::stream<ap_uint<{}>> &in1_V,
        hls::stream<ap_uint<{}>> &out0_V
    )""".format(
                        self.onnx_node.name,
                        self.get_instream_width(0),
                        wwidth,
                        self.get_outstream_width(),
                    )
                ]
                return

        else:
            raise Exception(
                """Please set mem_mode to "internal_embedded" or "internal_decoupled" or "external",
    currently no other parameter value is supported!"""
            )



    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        ram_style_thresholds = self.get_nodeattr("ram_style_thresholds")
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if mem_mode == "internal_embedded":
            self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
            # partition for parallel access along the PE dimension (dim 1)
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=weights.m_weights " "complete dim=1")
            )
        elif mem_mode == "internal_decoupled" or mem_mode == "external":
            # self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")
            # ------------------------------
            # derive to spmv implementation if sparse mode is set to 'spmv_sparse'
            try:
                sparse_mode = self.get_nodeattr("sparse_mode")
            except AttributeError:
                sparse_mode = "dense"
            is_spmv = (sparse_mode == "spmv_sparse")
            if is_spmv:
                self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=sfidx_V")
                self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=val_V")
                self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=mask_V")
                self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=rowlen_V")
            else:
                self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")

        else:
            raise Exception(
                """Please set mem_mode to "internal_embedded", "internal_decoupled", or external,
                currently no other parameter value is supported!"""
            )

        # the threshold tensor is acc_type [PE][TMEM][N_THRES]
        # partition for parallel access along PE and N_THRES
        # dimensions (dims 1 and 3)
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=1")
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=3")
            )
            # add resource pragma for thresholds if set
            if ram_style_thresholds == "distributed":
                self.code_gen_dict["$PRAGMAS$"].append(
                    ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_LUTRAM")
                )
            elif ram_style_thresholds == "block":
                self.code_gen_dict["$PRAGMAS$"].append(
                    ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_BRAM")
                )
            elif ram_style_thresholds == "auto":
                # no pragma needed
                pass
            else:
                raise Exception("Unrecognized ram_style_thresholds value:" + ram_style_thresholds)

    # def get_ap_int_max_w(self):
    #     # base class impl (max of inp/out stream widths)
    #     max_of_io = super().get_ap_int_max_w()
    #     # internal_decoupled mode weight stream
    #     weightstream = self.get_instream_width(1)
    #     simd = self.get_nodeattr("SIMD")
    #     if self.get_nodeattr("dynamic_input"):
    #         weightstream = weightstream * simd
    #     # single PE weight entry
    #     weight_bits = self.get_input_datatype(1).bitwidth()
    #     single_pe_w = simd * weight_bits
    #     return max([weightstream, max_of_io, single_pe_w])
    def get_ap_int_max_w(self):
        # 现有：考虑 in0_V/out0_V/... 的位宽
        maxw = max(self.get_instream_width(0), self.get_outstream_width())

        mem_mode = self.get_nodeattr("mem_mode")
        try:
            sparse_mode = self.get_nodeattr("sparse_mode")
        except AttributeError:
            sparse_mode = "dense"
        is_spmv = (sparse_mode == "spmv_sparse")

        if mem_mode in ["internal_decoupled", "external"]:
            if is_spmv:
                MW   = int(self.get_nodeattr("MW"))
                SIMD = int(self.get_nodeattr("SIMD"))
                PE   = int(self.get_nodeattr("PE"))
                import math
                ncols_per_lane = (MW + SIMD - 1) // SIMD
                sfidx_width = max(1, math.ceil(math.log2(ncols_per_lane)))
                wdt = self.get_input_datatype(1)
                export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
                wbits = export_wdt.bitwidth()

                cands = [
                    PE*SIMD*sfidx_width,  # sfidx
                    PE*SIMD*wbits,        # val
                    PE*SIMD,              # mask
                    PE*16,                # rowlen
                ]
                maxw = max([maxw] + cands)
            else:
                w = self.get_instream_width(1)
                if self.get_nodeattr("dynamic_input"):
                    w *= self.get_nodeattr("SIMD")
                maxw = max(maxw, w)
        return maxw


    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        dynamic_input = self.get_nodeattr("dynamic_input")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        # create a npy file fore each input of the node (in_ind is input index)
        for in_ind, inputs in enumerate(node.input):
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            assert (
                str(context[inputs].dtype) == "float32"
            ), """Input datatype is
            not float32 as expected."""

            if in_ind == 0:
                expected_inp_shape = self.get_folded_input_shape(in_ind)

                reshaped_input = context[inputs].reshape(expected_inp_shape)
                if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype(0)
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_0.npy"),
                    reshaped_input,
                )

            if in_ind == 1:
                if dynamic_input:
                    reshaped_input = context[inputs].reshape(-1, context[inputs].shape[-1])
                    self.make_weight_file(
                        reshaped_input, "decoupled_npy", "{}/input_1.npy".format(code_gen_dir)
                    )

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            assert (
                context[node.output[0]].shape == self.get_normal_output_shape()
            ), "cppsim did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
            self.reset_rtlsim(sim)

            if dynamic_input or mem_mode in ["external", "internal_decoupled"]:
                wnbits = self.get_instream_width(1)
                if self.get_nodeattr("dynamic_input"):
                    wnbits = wnbits * self.get_nodeattr("SIMD")
                export_wdt = self.get_input_datatype(1)

                # we have converted bipolar weights to binary for export,
                # so use it as such for weight generation
                if self.get_input_datatype(1) == DataType["BIPOLAR"]:
                    export_wdt = DataType["BINARY"]

                wei = npy_to_rtlsim_input("{}/input_1.npy".format(code_gen_dir), export_wdt, wnbits)
                num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))

                io_dict = {
                    "inputs": {"in0": inp, "in1": wei * num_w_reps},
                    "outputs": {"out0": []},
                }
            else:
                io_dict = {
                    "inputs": {"in0": inp},
                    "outputs": {"out0": []},
                }

            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)
            output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output_0.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def instantiate_ip(self, cmd):
        # instantiate the HLS IP
        vlnv = self.get_nodeattr("ip_vlnv")
        node_name = self.onnx_node.name
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            cmd.append("create_bd_cell -type ip -vlnv %s /%s/%s" % (vlnv, node_name, node_name))
        else:
            cmd.append("create_bd_cell -type ip -vlnv %s %s" % (vlnv, node_name))
