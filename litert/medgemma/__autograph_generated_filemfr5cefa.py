# coding=utf-8
def outer_factory():
    signature_function = None
    signature_key = None

    def inner_factory(ag__):

        def tf__signature_wrapper(**kwargs):
            with ag__.FunctionScope('signature_wrapper', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                structured_outputs = ag__.converted_call(ag__.ld(signature_function), (), dict(**ag__.ld(kwargs)), fscope)
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(_normalize_outputs), (ag__.ld(structured_outputs), ag__.ld(signature_function).name, ag__.ld(signature_key)), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__signature_wrapper
    return inner_factory