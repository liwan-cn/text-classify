#! /usr/bin/env python
import torch
import model
from model_util import *
from data_util import *
from args_util import *
import pickle

def main():
    args = get_args()
    # load data
    # args.predict = '希腊 月 神 毫米 双管 高炮法国 航宇 防务 近日 报道 美国 传出 空袭 伊朗核 设施 伊朗 备战 加快 伊朗 地面 构筑 多套 射程 防空 导弹 多层 防空网 伊朗 军方 放心 多次派 出国考察 各类 防空 武器 希腊 月 神 双管 高射炮 伊朗 军方 浓厚兴趣 伊朗 军方 希望 采购 一批 月 神 高射炮 保卫 伊朗核 设施 内层 防线 航宇 防务 称 这项 交易 难能 敲定 美国 北约 问月 神 希腊 国营 赫 伦尼 克 兵工厂 研制 年 服役 现 装备 希腊 国防军 塞浦路斯 国民 警卫队 用于 低空 防御 阻击 来袭 飞机 导弹 用于 对付 地面 装甲 目标 软 目标 月 神 牵引式 高炮 行军 状态 时炮 长米 高米 全重 吨 机动 能力 强 由名 炮手 操作 左炮 右炮 配名 装弹 手 另配 射手 名月 神 两管 口径 毫米 机关炮 安装 高 机动性 轮炮 架上 前视 红外 跟踪 装置 跟踪 雷达 自主 快速 探测 捕获 跟踪目标 高炮 采用 搜索 雷达 跟踪 雷达 相互 独立 一体 双 雷达 体制 提高 火控系统 快速 截获 目标 能力 计算机管理 自动化 操作系统 跟踪 天线 火炮 随动 系统 相结合 缩短 反应时间 系统 反应时间 小于 秒 月 神 多批 目标 快速 转换 攻击 能力 转换 时间 大于 秒月 神 牵引 高炮 采用 液压 驱动 自动 调平 装置 缩短 行军 状态 转为 状态 时间 双 炮管 一分钟 发射 发 炮弹 有效射程 公里 正好 伊朗 对空 防御 中 担心 薄弱环节 对付 空中 目标 目标 装甲 防护 情况 选用 榴弹 曳光 榴弹 曳光 穿甲 榴弹 对付 地面 装甲 目标 选用 穿甲 榴弹 脱壳 穿甲弹 各国 现役 小口径 高炮 相比 月 神 初速 高 空中 目标 杀伤 效果月 神 结构 简单 操作 维护 火炮 外形 低矮 隐蔽 模块 式 火控系统 灵活 利于 改进 昼夜 全天候 价格 便宜 每套 系统 包括 门 高炮 指挥 控制系统 载车 需 多万美元 报道 称 伊朗 打算 购得 月 神 安装 炮口 测速 装置 测速 装置 适时地 修正 弹丸 初速 变化 射击 偏差 季节 地理位置 气温 环境 条件 影响 初速 变化 多发 射击 身管 磨损 初速 变化 时 火炮 射击 精度 变化'
    if args.predict is None:
        if args.snapshot is None:
            # 首次训练, 加载词典
            word2id, id2word, label2id, id2label, file2label, max_len = process_data(args.file_path)

            if not os.path.isdir(args.vocab):
                os.makedirs(args.vocab)
            with open(args.vocab + '/' + 'word2id.pkl', 'wb') as f:
                print('Saving word2id...')
                pickle.dump(word2id, f)
            with open(args.vocab + '/' + 'id2label.pkl', 'wb') as f:
                print('Saving id2label...')
                pickle.dump(id2label, f)
            with open(args.vocab + '/' + 'file2label.pkl', 'wb') as f:
                print('Saving file2label...')
                pickle.dump(file2label, f)
        else:
            #继续训练, 加载词典
            with open(args.vocab + '/' + 'word2id.pkl', 'rb') as f:
                print('Loading word2id...')
                word2id = pickle.load(f)
            with open(args.vocab + '/' + 'id2label.pkl', 'rb') as f:
                print('Loading id2label...')
                id2label = pickle.load(f)
            with open(args.vocab + '/' + 'file2label.pkl', 'rb') as f:
                print('Loading file2label...')
                file2label = pickle.load(f)
        #print(id2label)
        # if args.max_len <= 0:
        #     args.max_len = max_len
        train_iter, test_iter = load_data(file2label, word2id, args)
    else:
        with open(args.vocab + '/' + 'word2id.pkl', 'rb') as f:
            print('Loading word2id...')
            word2id = pickle.load(f)
        with open(args.vocab + '/' + 'id2label.pkl', 'rb') as f:
            print('Loading id2label...')
            id2label = pickle.load(f)

    args.embed_num = len(word2id)
    args.class_num = len(id2label)

    print_parameters(args)
    # model
    cnn = model.CNN_Text(args)
    if args.snapshot is not None:
        print('\nLoading model from %s...' % (args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))

    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()

    # train or predict
    if args.predict is not None:
        print('\ntext:\n%s\nlabel:%s'
              % (args.predict, predict(args.predict, cnn, word2id, id2label, args.cuda)))
    else:
        print()
        try:
            train(train_iter, test_iter, cnn, args)
        except KeyboardInterrupt:
            print('Exiting from training early')

if __name__ == '__main__':
    main()