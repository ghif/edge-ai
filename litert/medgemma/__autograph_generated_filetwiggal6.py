# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf___get_shape_with_dynamic(signature: export.VariableSignature):
            with ag__.FunctionScope('_get_shape_with_dynamic', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = [ag__.if_exp(ag__.converted_call(ag__.ld(export_utils).is_torch_dynamic, (ag__.ld(s),), None, fscope), lambda: None, lambda: ag__.ld(s), 'ag__.converted_call(export_utils.is_torch_dynamic, (s,), None, fscope)') for s in ag__.ld(signature).shape]
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___get_shape_with_dynamic
    return inner_factory