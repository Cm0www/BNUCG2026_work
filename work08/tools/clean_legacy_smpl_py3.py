# -*- coding: utf-8 -*-
import argparse
import inspect
import os
import os.path as osp
import pickle
import numpy as np

# 兼容旧版 Chumpy 在新版 NumPy 中使用的已删除别名
np.bool = np.bool_
np.int = int
np.float = float
np.complex = complex
np.object = object
np.unicode = str
np.str = str
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: np.asarray(a).item()

# 兼容旧版 chumpy：Python 3.11+ 删除了 inspect.getargspec
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec


def is_chumpy(value):
    module = getattr(value.__class__, "__module__", "")
    return module == "chumpy" or module.startswith("chumpy.")


def clean_value(value):
    if is_chumpy(value):
        # Chumpy 对象的 .r 是实际数值结果
        try:
            return np.asarray(value.r)
        except Exception:
            return np.asarray(value)

    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean_value(v) for v in value]

    if isinstance(value, tuple):
        return tuple(clean_value(v) for v in value)

    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not osp.isfile(args.input):
        raise FileNotFoundError("找不到输入模型：" + args.input)

    output_dir = osp.dirname(osp.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print("读取旧模型：", args.input)
    with open(args.input, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    converted = []
    cleaned = {}

    for key, value in data.items():
        if is_chumpy(value):
            converted.append(key)
        cleaned[key] = clean_value(value)

    with open(args.output, "wb") as f:
        pickle.dump(cleaned, f, protocol=2)

    print("清理完成：", args.output)
    print("转换的 Chumpy 字段：", converted if converted else "未检测到顶层 Chumpy 字段")


if __name__ == "__main__":
    main()
