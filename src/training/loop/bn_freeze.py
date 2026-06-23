def set_frozen_modules_eval(model):
    """
    自动将所有完全冻结的子模块（其所有 requires_grad 参数为 False）设为 eval 模式，
    防止其 BatchNorm 更新运行统计数据。
    """
    for m in model.modules():
        params = list(m.parameters())
        if len(params) > 0 and all(not p.requires_grad for p in params):
            m.eval()
