# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__is_torch_dynamic(v):
            with ag__.FunctionScope('is_torch_dynamic', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(isinstance), (ag__.ld(v), ag__.ld(torch).SymInt), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__is_torch_dynamic
    return inner_factory