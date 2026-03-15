# coding=utf-8
def outer_factory():
    bundle = None
    exported_program = None
    tf_state_dict = None

    def inner_factory(ag__):

        def tf__inner(*args):
            with ag__.FunctionScope('inner', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                t_outs = [ag__.converted_call(ag__.ld(torch_dtype_to_tf), (ag__.ld(sig).dtype,), None, fscope) for sig in ag__.ld(bundle).output_signature]
                s_outs = [ag__.converted_call(ag__.ld(_get_shape_with_dynamic), (ag__.ld(sig),), None, fscope) for sig in ag__.ld(bundle).output_signature]
                call_args = ag__.converted_call(ag__.ld(_extract_call_args), (ag__.ld(bundle), ag__.ld(args), ag__.ld(tf_state_dict)), None, fscope)
                call_module_return = ag__.converted_call(ag__.ld(tfxla).call_module, (ag__.converted_call(ag__.ld(tuple), (ag__.ld(call_args),), None, fscope),), dict(version=5, Tout=ag__.ld(t_outs), Sout=ag__.ld(s_outs), function_list=[], module=ag__.ld(bundle).module_bytecode_vhlo), fscope)
                spec = ag__.ld(exported_program).call_spec.out_spec

                def get_state():
                    return (do_return, retval_)

                def set_state(vars_):
                    nonlocal retval_, do_return
                    do_return, retval_ = vars_

                def if_body():
                    nonlocal retval_, do_return
                    try:
                        do_return = True
                        retval_ = ag__.ld(call_module_return)
                    except:
                        do_return = False
                        raise

                def else_body():
                    nonlocal retval_, do_return
                    flat_names = ag__.converted_call(ag__.ld(common_utils).flat_dict_names, (ag__.ld(spec).children_specs, ag__.ld(spec).context), None, fscope)
                    try:
                        do_return = True
                        retval_ = {ag__.ld(name): ag__.ld(value) for name, value in ag__.converted_call(ag__.ld(zip), (ag__.ld(flat_names), ag__.ld(call_module_return)), None, fscope)}
                    except:
                        do_return = False
                        raise
                flat_names = ag__.Undefined('flat_names')
                ag__.if_stmt(ag__.not_(ag__.ld(spec).context), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
                return fscope.ret(retval_, do_return)
        return tf__inner
    return inner_factory