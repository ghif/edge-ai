# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__torch_dtype_to_tf(dtype):
            with ag__.FunctionScope('torch_dtype_to_tf', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = ag__.converted_call({ag__.ld(torch).double: ag__.ld(tf).float64, ag__.ld(torch).float32: ag__.ld(tf).float32, ag__.ld(torch).half: ag__.ld(tf).float16, ag__.ld(torch).long: ag__.ld(tf).int64, ag__.ld(torch).int32: ag__.ld(tf).int32, ag__.ld(torch).int16: ag__.ld(tf).int16, ag__.ld(torch).bool: ag__.ld(tf).bool, ag__.ld(torch).bfloat16: ag__.ld(tf).bfloat16}.get, (ag__.ld(dtype),), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__torch_dtype_to_tf
    return inner_factory