# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf___flatten_list(l: List) -> List:
            with ag__.FunctionScope('_flatten_list', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                flattened = []

                def get_state_1():
                    return ()

                def set_state_1(block_vars):
                    pass

                def loop_body(itr):
                    item = itr

                    def get_state():
                        return ()

                    def set_state(block_vars):
                        pass

                    def if_body():
                        ag__.converted_call(ag__.ld(flattened).extend, (ag__.converted_call(ag__.ld(_flatten_list), (ag__.ld(item),), None, fscope),), None, fscope)

                    def else_body():
                        ag__.converted_call(ag__.ld(flattened).append, (ag__.ld(item),), None, fscope)
                    ag__.if_stmt(ag__.converted_call(ag__.ld(isinstance), (ag__.ld(item), ag__.ld(list)), None, fscope), if_body, else_body, get_state, set_state, (), 0)
                item = ag__.Undefined('item')
                ag__.for_stmt(ag__.ld(l), None, loop_body, get_state_1, set_state_1, (), {'iterate_names': 'item'})
                try:
                    do_return = True
                    retval_ = ag__.ld(flattened)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___flatten_list
    return inner_factory