import torch

@torch.no_grad()
def get_type_scales(x, quant_values, sym=True, use_qset_minmax=False):
    # x: Shape (num_rows, num_elements)
    # quant_values: Shape (num_qsets, num_qvalues)
    num_qsets = quant_values.shape[0]
    num_rows = x.shape[0]
    x = x.unsqueeze(0)  # Shape: (1, num_rows, num_elements)
    quant_values = quant_values.unsqueeze(1)  # Shape: (num_qsets, 1, num_qvalues)

    if sym:
        if use_qset_minmax:
            qmax = quant_values.amax().repeat(num_qsets).unsqueeze(-1)  # Shape: (num_qsets, 1)
            qmin = quant_values.amin().repeat(num_qsets).unsqueeze(-1)  # Shape: (num_qsets, 1)
        else:
            qmax = quant_values.max(dim=2)[0]  # Shape: (num_qsets, 1)
            qmin = quant_values.min(dim=2)[0]  # Shape: (num_qsets, 1)

        xmax = x.amax(dim=2, keepdim=True)  # Shape: (1, num_rows, 1)
        xmin = x.amin(dim=2, keepdim=True)  # Shape: (1, num_rows, 1)

        xmax_abs = torch.max(xmax.abs(), xmin.abs())  # Shape: (1, num_rows, 1)
        q_range = torch.max(qmax.abs(), qmin.abs())   # Shape: (num_qsets, 1)

        scale = xmax_abs / q_range.unsqueeze(1)  # Shape: (num_qsets, num_rows, 1)
        scale = scale.clamp(min=1e-5, max=1e4)
        zero_point = torch.zeros_like(scale)
    else:
        #FIXME(brian1009): Not yet tested....
        # Asymmetric quantization (not used in this example)
        if use_qset_minmax:
            qmax = quant_values.amax().repeat(num_qsets).unsqueeze(-1)  # Shape: (num_qsets, 1)
            qmin = quant_values.amin().repeat(num_qsets).unsqueeze(-1)  # Shape: (num_qsets, 1)
        else:
            qmax = quant_values.max(dim=2)[0]  # Shape: (num_qsets, 1)
            qmin = quant_values.min(dim=2)[0]  # Shape: (num_qsets, 1)
        
        xmax = x.amax(dim=2, keepdim=True)  # Shape: (1, num_rows, 1)
        xmin = x.amin(dim=2, keepdim=True)  # Shape: (1, num_rows, 1)
        
        quant_values = quant_values - qmin  # Move to [0, qmax-qmin], Shape: (num_qsets, 1, num_qvalues)
        scale = (xmax - xmin) / (qmax - qmin)  # Shape: (num_qsets, num_rows, 1)
        scale = scale.clamp(min=1e-5, max=1e4)
        zero_point = -xmin / scale
        zero_point = zero_point.round()

    return scale, zero_point


@torch.no_grad()
def fake_quantize_with_type(x, scale, zero_point, quant_values, best_qset_indices=None, dtype=None):
    # x: Shape (num_rows, num_elements)
    # scale: Shape (num_qsets, num_rows, 1) or (num_rows, 1)
    # zero_point: Shape (num_qsets, num_rows, 1) or (num_rows, 1)
    # quant_values: Shape (num_qsets, num_qvalues)
    # best_qset_indices: None or Shape (num_rows)
    num_qsets, num_rows, num_elements = quant_values.shape[0], x.shape[0], x.shape[1]

    if dtype is None:
        dtype = x.dtype
    # HY: To save memory, we use float16 to compute distances for non-uniform quantization
    x = x.to(dtype)
    scale = scale.to(dtype)
    zero_point = zero_point.to(dtype)
    quant_values = quant_values.to(dtype)

    # HY: not sure why we need to unsqueeze here.
    #     This causes "shape mismatch" when we use two data types
    # if len(scale) == 2:
    #     assert len(scale) == len(zero_point)
    #     scale = scale.unsqueeze(0)
    #     zero_point = zero_point.unsqueeze(0)
    
    x = x.unsqueeze(0)  # Shape: (1, num_rows, num_elements)
    x = x / scale + zero_point  # Broadcasting, Shape: (num_qsets, num_rows, num_elements)
    
    min_values = quant_values.amin(dim=1, keepdim=True).unsqueeze(2)  # Shape: (num_qsets, 1, 1)
    max_values = quant_values.amax(dim=1, keepdim=True).unsqueeze(2)  # Shape: (num_qsets, 1, 1)
    x = x.clamp(min=min_values, max=max_values)
    
    # Compute absolute differences between x and quant_values
    x_flat = x.reshape(num_qsets, num_rows, num_elements, 1)  # Shape: (num_qsets, num_rows, num_elements, 1)
    quant_values = quant_values.unsqueeze(1).unsqueeze(2)     # Shape: (num_qsets, 1, 1, num_qvalues)
    abs_diff = (x_flat - quant_values).abs()  # Shape: (num_qsets, num_rows, num_elements, num_qvalues)
    idxs = abs_diff.argmin(dim=3)  # Shape: (num_qsets, num_rows, num_elements)

    # Do the rounding by indexing the quantization values with the indices from each sets
    quant_values = quant_values.squeeze(1).squeeze(1)  # Shape: (num_qsets, num_qvalues)
    idxs = idxs.reshape(idxs.shape[0], -1)  # Shape: (num_qsets, num_rows * num_elements)
    x_q = torch.gather(quant_values, 1, idxs)  # Shape: (num_qsets, num_rows * num_elements) 
    x_q = x_q.reshape(num_qsets, num_rows, num_elements)  # Shape: (num_qsets, num_rows, num_elements)
        
    x_dq = (x_q - zero_point) * scale  # Shape: (num_qsets, num_rows, num_elements)

    if best_qset_indices is not None:
        x_dq = x_dq[best_qset_indices, torch.arange(num_rows)]

    return x_dq




#################################  3-bit Datatypes  #################################
#INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
INT3 = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
#INT3 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SP_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_SP_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SM_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 5.0]
FP3_SM_NEG = [-5.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SR_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
FP3_SR_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


#################################  4-bit Datatypes  #################################
INT4 = [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] # 15 numbers
INT4_SCALED = [-16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0] # 15 numbers
FLINT4 = [-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0] # 15 numbers

FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0] # 15 numbers
FP4_SP_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0] # 16 numbers
FP4_SP_NEG = [-12.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0] # 16 numbers

FP4_SM_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0] # 16 numbers
FP4_SM_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0] # 16 numbers

FP4_SR_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0] # 16 numbers
FP4_SR_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0] # 16 numbers

APOT4 = [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0] # 15 numbers
APOT4_SP_POS = [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] # 16 numbers
APOT4_SP_NEG = [-10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0] # 16 numbers

APOT4_SR_POS = [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 16.0] # 16 numbers
APOT4_SR_NEG = [-16.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0] # 16 numbers


#################################  5-bit Datatypes  #################################
INT5 = [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
FLINT5 = [-64.0, -32.0, -24.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 32.0, 64.0]
FP5_E2M2 = [-28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0]
FP5_E3M1 = [-192.0, -128.0, -96.0, -64.0, -48.0, -32.0, -24.0, -16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]


DATATYPE_MAPPING_3_BIT = {'int3': INT3, 'fp3': FP3, 
                        'fp3_sp_pos': FP3_SP_POS, 'fp3_sp_neg': FP3_SP_NEG, 
                        'fp3_sm_pos': FP3_SM_POS, 'fp3_sm_neg': FP3_SM_NEG, 
                        'fp3_sr_pos': FP3_SR_POS, 'fp3_sr_neg': FP3_SR_NEG, }

DATATYPE_MAPPING_4_BIT = {'int4': INT4, 'int4_scaled': INT4_SCALED, 'fp4': FP4_E2M1, 'flint4': FLINT4,
                        'fp4_sp_pos': FP4_SP_POS, 'fp4_sp_neg': FP4_SP_NEG, 
                        'fp4_sm_pos': FP4_SM_POS, 'fp4_sm_neg': FP4_SM_NEG, 
                        'fp4_sr_pos': FP4_SR_POS, 'fp4_sr_neg': FP4_SR_NEG, 
                        'apot4': APOT4,
                        'apot4_sp_pos': APOT4_SP_POS, 'apot4_sp_neg': APOT4_SP_NEG, 
                        'apot4_sr_pos': APOT4_SR_POS, 'apot4_sr_neg': APOT4_SR_NEG,                         
                        }

DATATYPE_MAPPING_5_BIT = {'int5': INT5, 'fp5': FP5_E2M2, 'flint5': FLINT5,
                        'fp5_e2m2': FP5_E2M2, 'fp5_e3m1': FP5_E3M1}

ALL_DATATYPES = {**DATATYPE_MAPPING_3_BIT, **DATATYPE_MAPPING_4_BIT, **DATATYPE_MAPPING_5_BIT}


def get_datatypes(datatype: str, wq_bits: int):
    if wq_bits == 3:
        if datatype == 'int':
            datatype_list = ['int3']
        elif datatype == 'fp':
            datatype_list = ['fp3']
        elif datatype == 'mixed':
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg', 'fp3_sr_pos', 'fp3_sr_neg']
        elif datatype == 'mixed_int':
            datatype_list = ['int3', 'fp3_sp_pos', 'fp3_sp_neg', 'fp3_sr_pos', 'fp3_sr_neg']
        elif datatype == 'mixed_sp':
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg']
        elif datatype == 'mixed_sm':
            datatype_list = ['fp3_sm_pos', 'fp3_sm_neg']
        elif datatype == 'mixed_sr':
            datatype_list = ['fp3_sr_pos', 'fp3_sr_neg']
        elif datatype == 'mixed_sp_sm':
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg', 'fp3_sm_pos', 'fp3_sm_neg']
        elif datatype == 'mixed_sm_sr':
            datatype_list = ['fp3_sm_pos', 'fp3_sm_neg', 'fp3_sr_pos', 'fp3_sr_neg']
        elif datatype == 'mixed_sp_sr':
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg', 'fp3_sr_pos', 'fp3_sr_neg']
        else:
            raise ValueError(f"Unexpected datatype: {datatype}")
    elif wq_bits == 4:
        if datatype == 'int':
            datatype_list = ['int4']
        elif datatype == 'fp':
            datatype_list = ['fp4']
        elif datatype == 'mixed':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg', 'fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_int':
            datatype_list = ['int4', 'fp4_sp_pos', 'fp4_sp_neg', 'fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_sp':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg']
        elif datatype == 'mixed_sm':
            datatype_list = ['fp4_sm_pos', 'fp4_sm_neg']
        elif datatype == 'mixed_sr':
            datatype_list = ['fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_sp_sm':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg', 'fp4_sm_pos', 'fp4_sm_neg']
        elif datatype == 'mixed_sm_sr':
            datatype_list = ['fp4_sm_pos', 'fp4_sm_neg', 'fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_sp_sr':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg', 'fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_apot4':
            datatype_list = ['apot4_sp_pos', 'apot4_sp_neg', 'apot4_sr_pos', 'apot4_sr_neg']
        else:
            raise ValueError(f"Unexpected datatype: {datatype}")
    elif wq_bits == 5:
        if datatype == 'int':
            datatype_list = ['int5']
        elif datatype == 'fp':
            datatype_list = ['fp5']
        elif datatype == 'mixed':
            datatype_list = ['fp5_e2m2', 'fp5_e3m1']
        elif datatype == 'mixed_int':
            datatype_list = ['int5', 'fp5_e2m2', 'fp5_e3m1']

    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit and 5-bit quantization, not {wq_bits}-bit")
    
    return datatype_list

def get_quant_value_from_dtype_lists(datatype_list):
    quant_values_list = []
    for datatype in datatype_list:
        if datatype not in ALL_DATATYPES:
            raise ValueError(f"Unsupported datatype: {datatype}")
        quant_values_list.append(ALL_DATATYPES[datatype])
    return quant_values_list