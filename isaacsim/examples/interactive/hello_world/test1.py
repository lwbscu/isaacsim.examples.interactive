# 简化的Isaac Sim连接测试
import omni.kit
import omni.usd

print("成功连接到Isaac Sim!")
print("Isaac Sim版本:", omni.kit.app.get_app().get_build_version())

# 获取当前场景信息
stage = omni.usd.get_context().get_stage()
if stage:
    print("当前场景已加载")
    # 打印一些基本场景信息
    print("场景路径:", stage.GetRootLayer().identifier)
    print("场景中的基本元素:")
    count = 0
    for prim in stage.Traverse():
        if count < 5:  # 只显示前5个，避免信息过多
            print(f"- {prim.GetPath()}")
        count += 1
    print(f"总计: {count}个元素")
else:
    print("当前没有加载场景")

print("连接测试成功完成!")