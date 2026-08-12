import torch


def gate8_adaptor_isolation(model, criterion, x, targets):
    """TaskAlignmentAdaptor cls→seg Gradient Isolation Audit.

    审计契约(`multi_task_model.py` 的 `_forward_cls_backbone`,不写行号——该行号已因
    无关改动过时两次,符号名比行号耐改):
        x_cls = self.adaptor_stage3(seg_features[1].detach(), x_cls)
        x_cls = self.adaptor_stage4(seg_features[2].detach(), x_cls)
    `seg_features` 必须先 `.detach()` 再喂给 adaptor,防止分类误差梯度污染分割主干
    `seg_backbone`。既有 7 个 Gate 都测不到这条契约:
      - gate2_vclub 测的是 `z_latent.detach()` 之后的 vCLUB 局部图,根本不经过
        TaskAlignmentAdaptor 的计算路径,结构上必然通过,形同虚设。
      - gate3_auc_margin 的 Rule 3.3 会把 `seg_backbone.requires_grad` 显式设为
        False(用于其后的 BatchNorm 冻结校验),这个状态会残留在共享 `model` 实例上。
        若不显式复位,即便 `.detach()` 被误删,`seg_backbone` 也照样收不到梯度
        (requires_grad=False 直接短路 autograd)——用"假冻结"掩盖了"detach 失效"。
      - gate6_stage1_masking 测的是 `enable_mal_loss=False` 时 `loss_mal` 被整体
        乘 0,跟 detach 契约无关(乘 0 本身就会让梯度恒为 0,测不出 detach 是否失效)。
    """
    orig_gamma3 = None
    orig_gamma4 = None
    # 2026-07-11 Gate2 review 非阻塞建议:之前只在 finally 里复原 gamma,没复原
    # requires_grad——当时因为 Gate8 是流水线最后一个 Gate 所以无实际影响,但这是
    # 对"Gate8 排最后"这个顺序假设的隐性依赖,未来插入 Gate9 若依赖干净的
    # requires_grad 状态会踩坑。改为显式记录原值并在 finally 里逐参数复原。
    orig_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        # ── 前置状态复位:不依赖/不污染其他 Gate 的执行顺序 ──────────────────
        # 1) requires_grad 必须显式重置为 True,理由见上方 docstring 的 gate3 一条:
        #    这正是"之前的 Gate 会掩盖该问题"的具体机制,必须在本 Gate 内主动修正。
        for p in model.parameters():
            p.requires_grad = True

        # 2) TaskAlignmentAdaptor.gamma 初始化为 nn.Parameter(torch.zeros(1))
        #    ("零初始化残差门控",让 adaptor 训练早期从恒等映射开始学习,是设计本身,
        #    不是 bug)。verify_model.py 每次都新建 model(pretrained=False),此时
        #    gamma 恒为 0 ——`f_cls + gamma*(f_cls*mask)` 里的注入项数值上恒为 0,
        #    导致无论 `.detach()` 是否还在,这条通路对 f_seg 的雅可比都被 gamma=0
        #    这个乘法因子清零,和"detach 生效"是两回事,却会产生一模一样的
        #    "grad=0"表象——即"gamma=0 时不管有没有 detach,测出来都是 0"。
        #    已用脚本对拍实测验证(简报附数据):gamma=0 时故意 monkeypatch 掉
        #    `torch.Tensor.detach`(模拟"误删 detach"),`seg_backbone` 梯度依然精确
        #    为 0.0 —— 说明不强制 gamma≠0 的版本是一个永远通过的伪 Gate,起不到
        #    任何审计作用。因此必须显式把 gamma 打到非零值,让注入通路数值上真正激活。
        orig_gamma3 = model.adaptor_stage3.gamma.detach().clone()
        orig_gamma4 = model.adaptor_stage4.gamma.detach().clone()
        with torch.no_grad():
            model.adaptor_stage3.gamma.fill_(1.0)
            model.adaptor_stage4.gamma.fill_(1.0)

        model.train()
        model.zero_grad()

        # 3) gamma_prior=0.0 精确清零"物理先验通路"对 loss_mal 的合法贡献。
        #    cbm_head.forward 里 `c_fused = cat([c_flat, e_prior])`,而
        #    `e_prior = prior_mlp(v_phys) * gamma_prior`,`v_phys` 来自
        #    `prior_extractor(soft_mask, x)`,`soft_mask` 来自 `seg_head(d)`——
        #    `d` 由 `dec_stages`/`up_stages` 消费**未 detach** 的 `seg_features`
        #    计算而得。这是设计内的合法耦合(物理先验反哺分类分支,不是要审计的缺口),
        #    但若不消掉它,`loss_mal` 反向传播天然会把非零梯度灌回 `seg_backbone`——
        #    已实测:`gamma_prior=1.0`(默认值)下 `seg_backbone` 梯度
        #    `abs().sum()≈14.2`,且与是否删除 adaptor 的 `.detach()` 完全无关,
        #    若不排除会把这条无关合法通路误判成"泄漏"(等同噪声超过信号,Gate 恒假阳)。
        #    `gamma_prior` 是 Python float 字面量 0.0,IEEE754 下 `finite*0.0` 精确
        #    为 0(无舍入误差),该乘法节点会把回传到此处的梯度精确清零,并把精确 0
        #    一路传播穿过后面所有线性算子(Conv/BN/ConvTranspose/Sigmoid——见
        #    §末尾断言阈值说明)直到 `seg_backbone`。消掉该通路后,
        #    `loss_mal → seg_backbone` 唯一存活的路径就只剩 TaskAlignmentAdaptor
        #    的两个 injection 点,精确对应本 Gate 要审计的契约。
        out = model(x, gamma_prior=0.0)
        _, loss_dict = criterion(out, targets, update_famo_weights=False)
        loss_mal = loss_dict["malignancy"]
        loss_mal.backward()

        # ── 断言:seg_backbone 收到的梯度必须精确为 0,而非"很小" ────────────
        # 理由(不照抄 QA_VP.md §1.1 模板的 1e-7 容差,原因如下):
        #   上面两步(detach 图断开 + gamma_prior=0 乘法清零)在数学上都不是
        #   "约等于 0"而是"精确等于 0"——`.detach()` 是计算图层面的硬断开
        #   (该路径根本不存在,不是"存在但很小");`gamma_prior=0.0` 是字面浮点数
        #   乘法,`0.0*finite=0.0` 精确成立,且对 Conv/BN/Linear 等线性算子的
        #   反传逐层保持精确 0(见上方 3. 的推导)。因此用 `!=0.0`(而非 `>1e-7`)：
        #     1. 数学上更严格,能捕捉比 1e-7 更小的意外泄漏;
        #     2. `!=` 对 NaN 天然敏感——若梯度意外产生 NaN(数值健康问题),
        #        `float('nan') != 0.0` 为 True,会被判定为 leaked 从而 fail-loud;
        #        而 `>1e-7`/`>0.0` 式判断对 NaN 有盲区(`NaN > x` 恒为 False,
        #        会静默放过,是 gate2/gate6 现有写法的一个隐藏盲点,本 Gate 不重蹈);
        #     3. 与 gate6_stage1_masking 已有的 `>0.0`(无容差)风格同源——两者
        #        都是"乘 0 恒为精确 0"的场景,本 Gate 属于同一类,只是外加了
        #        detach 图断开这另一重精确为 0 的保证。
        #   已用脚本双向对拍实测(简报附数据):detach 正确时 `abs().sum()==0.0`;
        #   monkeypatch `torch.Tensor.detach` 为恒等函数模拟"误删 detach"时,
        #   `abs().sum()≈6094`(远超 1e-7 量级),两种场景界限分明,无需容差空间。
        leaked = False
        leaked_details = []
        for name, p in model.seg_backbone.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() != 0.0:
                leaked = True
                leaked_details.append(
                    f"    '{name}': grad_abs_sum={p.grad.abs().sum().item()}"
                )

        assert not leaked, (
            "TaskAlignmentAdaptor cls→seg 梯度隔离被破坏:loss_mal 反向传播后 "
            "seg_backbone 收到非零梯度,说明 multi_task_model.py 的 "
            "`seg_features[1].detach()`/`seg_features[2].detach()` 缺失或失效!\n"
            + "\n".join(leaked_details)
        )
        print(
            "✔ GATE 8: TaskAlignmentAdaptor cls→seg gradient isolation verified "
            "(loss_mal produces exactly-zero gradient on seg_backbone)."
        )
        return True
    except Exception as e:
        print(f"✘ GATE 8 Failed: {e}")
        return False
    finally:
        # 恢复 gamma 原值,避免污染后续流程(即便当前是最后一个 Gate,也保持
        # "每个 Gate 不留副作用"的洁癖,防止未来插入新 Gate 时踩坑)。
        if orig_gamma3 is not None:
            with torch.no_grad():
                model.adaptor_stage3.gamma.copy_(orig_gamma3)
        if orig_gamma4 is not None:
            with torch.no_grad():
                model.adaptor_stage4.gamma.copy_(orig_gamma4)
        # 同理复原 requires_grad,不只复原 gamma(见上方 2026-07-11 注释)。
        for name, p in model.named_parameters():
            if name in orig_requires_grad:
                p.requires_grad = orig_requires_grad[name]
