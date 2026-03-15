# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__flat_dict_names(tree_spec: pytree.TreeSpec, context: pytree.Context) -> List[str]:
            """Given a TreeSpec, this produces a list of names for the leaves.

  The list of names embeddeds the structure of the tree_spec. A nesting level is
  indicated by an `_` and elements in a list are indicated by `_<index>`.

  TODO b/361601485: The flattening of names is not collision-free and needs to
  be revised.

  Args:
    tree_spec: The TreeSpec to extract the names from.
    context: The context used to check if the provided spec belongs to a
      dictionary or a list.

  Returns:
    A list of flattened names.
  """
            with ag__.FunctionScope('flat_dict_names', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                flat_names = []

                def get_state_4():
                    return ()

                def set_state_4(block_vars):
                    pass

                def if_body_2():

                    def get_state_1():
                        return ()

                    def set_state_1(block_vars):
                        pass

                    def loop_body(itr):
                        i, spec = itr

                        def get_state():
                            return ()

                        def set_state(block_vars):
                            pass

                        def if_body():
                            ag__.converted_call(ag__.ld(flat_names).extend, ([f'{ag__.ld(i)}_{ag__.ld(name)}' for name in ag__.converted_call(ag__.ld(flat_dict_names), (ag__.ld(spec).children_specs, ag__.ld(spec).context), None, fscope)],), None, fscope)

                        def else_body():
                            ag__.converted_call(ag__.ld(flat_names).append, (f'{ag__.ld(i)}',), None, fscope)
                        ag__.if_stmt(ag__.ld(spec).children_specs, if_body, else_body, get_state, set_state, (), 0)
                    spec = ag__.Undefined('spec')
                    i = ag__.Undefined('i')
                    ag__.for_stmt(ag__.converted_call(ag__.ld(enumerate), (ag__.ld(tree_spec),), None, fscope), None, loop_body, get_state_1, set_state_1, (), {'iterate_names': '(i, spec)'})

                def else_body_2():
                    flat_ctx = ag__.converted_call(ag__.ld(_flatten_list), (ag__.ld(context),), None, fscope)

                    def get_state_3():
                        return ()

                    def set_state_3(block_vars):
                        pass

                    def loop_body_1(itr_1):
                        prefix, spec = itr_1
                        leaf_flat_names = ag__.converted_call(ag__.ld(flat_dict_names), (ag__.ld(spec).children_specs, ag__.ld(spec).context), None, fscope)

                        def get_state_2():
                            return ()

                        def set_state_2(block_vars):
                            pass

                        def if_body_1():
                            ag__.converted_call(ag__.ld(flat_names).extend, ([f'{ag__.ld(prefix)}_{ag__.ld(name)}' for name in ag__.ld(leaf_flat_names)],), None, fscope)

                        def else_body_1():
                            ag__.converted_call(ag__.ld(flat_names).append, (ag__.ld(prefix),), None, fscope)
                        ag__.if_stmt(ag__.ld(leaf_flat_names), if_body_1, else_body_1, get_state_2, set_state_2, (), 0)
                    leaf_flat_names = ag__.Undefined('leaf_flat_names')
                    prefix = ag__.Undefined('prefix')
                    spec = ag__.Undefined('spec')
                    ag__.for_stmt(ag__.converted_call(ag__.ld(zip), (ag__.ld(flat_ctx), ag__.ld(tree_spec)), None, fscope), None, loop_body_1, get_state_3, set_state_3, (), {'iterate_names': '(prefix, spec)'})
                prefix = ag__.Undefined('prefix')
                i = ag__.Undefined('i')
                leaf_flat_names = ag__.Undefined('leaf_flat_names')
                flat_ctx = ag__.Undefined('flat_ctx')
                spec = ag__.Undefined('spec')
                ag__.if_stmt(ag__.ld(context) is None, if_body_2, else_body_2, get_state_4, set_state_4, (), 0)
                try:
                    do_return = True
                    retval_ = ag__.ld(flat_names)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__flat_dict_names
    return inner_factory