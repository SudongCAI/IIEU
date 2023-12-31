import math

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=75, lr_step_soft=30, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.lr_step_soft = lr_step_soft
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'linear':
            lr = self.lr * (1 - 1.0 * T / self.N)
        elif self.mode == 'cooldown':
            lr = 1e-05
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == 'step_soft':
            lr = self.lr * (0.5 ** (epoch // self.lr_step_soft))
        elif self.mode == 'step_soft_cos':
            epoch_tmp = epoch % self.lr_step_soft
            self.N = self.lr_step_soft * self.iters_per_epoch # num of lr changings in each step
            T = epoch_tmp * self.iters_per_epoch + i 
            lr = 0.5 * (self.lr * (0.5 ** (epoch // self.lr_step_soft)) ) + \
                0.5 * (self.lr * (0.5 ** (epoch // self.lr_step_soft)) ) * (0.5 * (1 + math.cos(1.0 * T / self.N * math.pi)))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr, epoch)

    def _adjust_learning_rate(self, optimizer, lr, epoch):
        # all in
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr
        
        lr = lr if lr > 1e-05 else 1e-05 # default: open
        
        if len(optimizer.param_groups) == 25:
            # enlarge the lr at the head
            for i in range(0, 8):
                optimizer.param_groups[i]['lr'] = lr
            
            for i in range(8, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            

        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                

