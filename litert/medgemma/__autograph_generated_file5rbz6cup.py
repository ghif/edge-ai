# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf___extract_call_args(bundle: export.MlirLowered, args: Tuple[Any], tf_state_dict: Dict[str, tf.Variable]):
            with ag__.FunctionScope('_extract_call_args', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                call_args = []

                def get_state_2():
                    return ()

                def set_state_2(block_vars):
                    pass

                def loop_body(itr):
                    sig = itr

                    def get_state_1():
                        return ()

                    def set_state_1(block_vars):
                        pass

                    def if_body_1():
                        ag__.converted_call(ag__.ld(call_args).append, (ag__.ld(args)[ag__.ld(sig).input_spec.i],), None, fscope)

                    def else_body_1():

                        def get_state():
                            return ()

                        def set_state(block_vars):
                            pass

                        def if_body():
                            name = ag__.ld(sig).input_spec.name
                            ag__.converted_call(ag__.ld(call_args).append, (ag__.ld(tf_state_dict)[ag__.ld(name)],), None, fscope)

                        def else_body():
                            pass
                        ag__.if_stmt(ag__.ld(sig).input_spec.is_parameter, if_body, else_body, get_state, set_state, (), 0)
                    ag__.if_stmt(ag__.ld(sig).input_spec.is_user_input, if_body_1, else_body_1, get_state_1, set_state_1, (), 0)
                sig = ag__.Undefined('sig')
                name = ag__.Undefined('name')
                ag__.for_stmt(ag__.ld(bundle).input_signature, None, loop_body, get_state_2, set_state_2, (), {'iterate_names': 'sig'})
                try:
                    do_return = True
                    retval_ = ag__.ld(call_args)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___extract_call_args
    return inner_factory