import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from functools import partial

import numpy as np
import tensorflow as tf

train_gpu       = [0,]
os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
ngpus_per_node                      = len(train_gpu)
    
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
        
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.facenet import facenet
# mobilenet
from nets.facenet import face_analysis_model

from nets.facenet_training import get_lr_scheduler, triplet_loss
from utils.callbacks import (ExponentDecayScheduler, LFW_callback, LossHistory,
                             ModelCheckpoint)
from utils.dataloader import FacenetDataset, LFWDataset
from utils.utils import get_num_classes, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用eager模式训练
    #----------------------------------------------------#
    eager           = False
    #---------------------------------------------------------------------#
    #   train_gpu   训练用到的GPU
    #               默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
    #               在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #--------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    #--------------------------------------------------------#
    annotation_path = "cls_train.txt"
    #--------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    #--------------------------------------------------------#
    input_shape     = [160, 160, 3]
    #--------------------------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------------------------#
    
    # ***init****
    # backbone        = "mobilenet"
    
    # ***test***
    backbone        = "inception_resnetv2"
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    
    # ***init**
    model_path      = r"C:\Users\606\Desktop\lab606\logs\facenet_normal\ep100-loss0.007-val_loss0.531.h5"
    # model_path = "model_data/facenet_keras_weights.h5"
    
    # ***test***
    # model_path      = r"C:\Users\606\Desktop\facenet0523\facenet-tf2-main\logs\loss_2022_06_22_17_00_37\ep050-loss0.007-val_loss0.838.h5"
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，不能为1。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   Epoch           模型总共训练的epoch
    #   batch_size      每次输入的图片数量
    #                   受到数据加载方式与triplet loss的影响
    #                   batch_size需要为3的倍数
    #------------------------------------------------------#
    Init_Epoch      = 0
    Epoch           = 10
    batch_size      = 3

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-6
    Min_lr              = Init_lr * 0.01
    # Min_lr = Init_lr
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者1 
    #------------------------------------------------------------------#
    num_workers     = 1
    #------------------------------------------------------------------#
    #   是否开启LFW评估
    #------------------------------------------------------------------#
    lfw_eval_flag   = False
    #------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    #------------------------------------------------------------------#
    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "model_data/lfw_pair.txt"

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    
    
    #------------------------------------------------------#
    #   判断当前使用的GPU数量与机器上实际的GPU数量
    #------------------------------------------------------#
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")

    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))

    num_classes = get_num_classes(annotation_path)
    
    if backbone == "mobilenet":
        if ngpus_per_node > 1:
            with strategy.scope():
                #---------------------------------#
                #   载入模型并加载预训练权重
                #---------------------------------#
                model = facenet(input_shape, num_classes, backbone=backbone, mode="train")
                model.summary()
                if model_path != '':
                    #---------------------------------#
                    #   载入预训练权重
                    #---------------------------------#
                    model.load_weights(model_path, by_name=True, skip_mismatch=True)
        else:
            #---------------------------------#
            #   载入模型并加载预训练权重
            #---------------------------------#
            model = facenet(input_shape, num_classes, backbone=backbone, mode="train")
            model.summary()
            if model_path != '':
                #---------------------------------#
                #   载入预训练权重
                #---------------------------------#
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
            
    if backbone == "inception_resnetv2":
        if ngpus_per_node > 1:
            with strategy.scope():
                #---------------------------------#
                #   载入模型并加载预训练权重
                #---------------------------------#
                model = facenet(input_shape, num_classes, backbone=backbone, mode="train")
                model.summary()
                if model_path != '':
                    #---------------------------------#
                    #   载入预训练权重
                    #---------------------------------#
                    model.load_weights(model_path, by_name=True, skip_mismatch=True)
        else:
            #---------------------------------#
            #   载入模型并加载预训练权重
            #---------------------------------#
            model = facenet(input_shape, num_classes, backbone=backbone, mode="train")
            model.summary()
            if model_path != '':
                #---------------------------------#
                #   载入预训练权重
                #---------------------------------#
                model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #-------------------------------------------------------#
    val_split = 0.05
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )    

    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataset   = FacenetDataset(input_shape, lines[:num_train], batch_size, num_classes, ngpus_per_node, random = True)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], batch_size, num_classes, ngpus_per_node, random = False)

        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]

        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataset.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataset.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            
            if ngpus_per_node > 1:
                gen     = strategy.experimental_distribute_dataset(gen)
                gen_val = strategy.experimental_distribute_dataset(gen_val)

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)
            #---------------------------------#
            #   LFW估计
            #---------------------------------#
            if lfw_eval_flag:
                test_loader = LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, batch_size=32, input_shape=input_shape).generate()
            else:
                test_loader = None
                
            for epoch in range(Init_Epoch, Epoch):
                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)
                
                fit_one_epoch(model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            Epoch, triplet_loss(), test_loader, lfw_eval_flag, save_period, save_dir, strategy)

                train_dataset.on_epoch_end()
                val_dataset.on_epoch_end()
        
        else:
            if ngpus_per_node > 1:
                with strategy.scope():
                    model.compile(
                        # loss={'Embedding' : triplet_loss(), 'Softmax' : 'categorical_crossentropy'}, 
                        loss={'Softmax' : 'categorical_crossentropy'},
                        optimizer = optimizer, metrics = {'Softmax' : 'categorical_accuracy'}
                    )
            else:
                model.compile(
                    # loss={'Embedding' : triplet_loss(), 'Softmax' : 'categorical_crossentropy'}, 
                    loss={'Softmax' : 'categorical_crossentropy'},
                    optimizer = optimizer, metrics = {'Softmax' : 'categorical_accuracy'}
                )
                
            #-------------------------------------------------------------------------------#
            #   训练参数的设置
            #   logging         用于设置tensorboard的保存地址
            #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
            #   lr_scheduler       用于设置学习率下降的方式
            #   early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
            #-------------------------------------------------------------------------------#
            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            logging         = TensorBoard(log_dir)
            loss_history    = LossHistory(log_dir)
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            #---------------------------------#
            #   LFW估计
            #---------------------------------#
            if lfw_eval_flag:
                lfw_callback    = LFW_callback(LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, batch_size=32, input_shape=input_shape))
                callbacks       = [logging, loss_history, checkpoint, lr_scheduler, lfw_callback]
            else:
                callbacks       = [logging, loss_history, checkpoint, lr_scheduler]

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit(
                x                   = train_dataset,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataset,
                validation_steps    = epoch_step_val,
                epochs              = Epoch,
                initial_epoch       = Init_Epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
