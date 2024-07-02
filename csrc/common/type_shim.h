#include <ATen/ATen.h>

#define DISPATCH_ITYPE_INTEGRAL(ITYPE, NAME, ...)                                   \
    if(ITYPE == at::ScalarType::Char) {                                             \
        using input_t = int8_t;                                                     \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_INTEGRAL(WTYPE, NAME, ...)                                    \
    if (WTYPE == at::ScalarType::Char) {                                             \
       using weight_t = int8_t;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }


// #define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
//     if (ITYPE == at::ScalarType::Half) {                                            \
//         using input_t = at::Half;                                                   \
//         __VA_ARGS__();                                                              \
//     } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
//         using input_t = at::BFloat16;                                               \
//         __VA_ARGS__();                                                              \
//     } else if (ITYPE == at::ScalarType::Float)  {                                   \
//         using input_t = float;                                                      \
//         __VA_ARGS__();                                                              \
//     } else {                                                                        \
//         AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
//     }

// #define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
//     if (WTYPE == at::ScalarType::Half) {                                             \
//         using weight_t = at::Half;                                                   \
//         __VA_ARGS__();                                                               \
//     } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
//         using weight_t = at::BFloat16;                                               \
//         __VA_ARGS__();                                                               \
//     } else if (WTYPE == at::ScalarType::Float)  {                                    \
//         using weight_t = float;                                                      \
//         __VA_ARGS__();                                                               \
//     } else {                                                                         \
//         AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
//     }


#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, NAME, ...)         \
    if (TYPEIN == at::ScalarType::Double) {                                          \
       using scalar_t_in = double;                                                   \
       using scalar_t_out = int8_t;                                                  \
        __VA_ARGS__();                                                               \
    } else if (TYPEIN == at::ScalarType::Float) {                                    \
       using scalar_t_in = float;                                                    \
       using scalar_t_out = int8_t;                                                  \
        __VA_ARGS__();                                                               \
    } else if (TYPEIN == at::ScalarType::Half) {                                     \
       using scalar_t_in = at::Half;                                                 \
       using scalar_t_out = int8_t;                                                  \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");            \
    }
