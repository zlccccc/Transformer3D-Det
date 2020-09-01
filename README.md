## training codes

You Should download thop and change thop/profile.py line 192 and 206(dfs\_count)
    print(prefix, module.\_get\_name(), ':')
    print(prefix, module.\_get\_name(), clever_format([total_ops, total_params], '%.3f'), flush=True)
