'''
mobilenet_v2
CIARD
push according to label
consist are decided by top1 prediction
Lr stage decay
'''
import os
import torch
from mtard_loss import *
from cifar10_models import *
from cifar10_nat_teacher_models import *
import torchvision
from torchvision import transforms
from loguru import logger
import math
# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

prefix = 'Cifar10_MobileNetV2'
draw_file = prefix
model_dir = './model/' + prefix
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

with open('./model/' + prefix+ '/'+ draw_file,'w') as f:
    text = "epoch student_robust_acc student_natural_acc adv_teacher_robust_acc adv_teacher_natural_acc nat_teacher_robust_acc nat_teacher_natural_acc\n"
    f.write(text)
epochs = 300
batch_size = 128
epsilon = 8/255.0
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

student = mobilenet_v2()#resnet18() 
resume_student_path = None 
if resume_student_path != None:
    state_dict = torch.load(resume_student_path,map_location=torch.device('cpu'))["model"]
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    student.load_state_dict(new_state_dict)
student = student.cuda()
student.train()
if(resume_student_path == None):
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
else:
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

begin_epoch = 1 if resume_student_path == None else 200

weight = {
    "adv_loss": 1/2.0,
    "nat_loss": 1/2.0,
}
init_loss_nat = None
init_loss_adv = None



def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss

def entropy_value(a):
    value = torch.log(a+1e-5)*a
    return value

def scale_to_magnitude(a, b, c):
    if(math.isclose(a, 0, rel_tol=1e-9)): a += 1e-7
    if(math.isclose(b, 0, rel_tol=1e-9)): b += 1e-7
    if(math.isclose(c, 0, rel_tol=1e-9)): c += 1e-7
    magnitude_a = math.floor(math.log10(abs(a)))
    magnitude_b = math.floor(math.log10(abs(b)))
    target_magnitude = min(magnitude_a , magnitude_b)
    magnitude_c = math.floor(math.log10(abs(c)))
    scale_factor = 10 ** (target_magnitude - magnitude_c)
    scaled_c = scale_factor #*c
    return scaled_c

def push_loss(teacher_logits, students_logits, labels,T = 5):#train_batch_labels
    '''print(teacher_logits.shape)
    print(students_logits.shape)
    print(labels.shape)'''
    teacher_predictions = torch.argmax(teacher_logits, dim=1)
    #print(teacher_predictions.shape)
    diff_indices = (teacher_predictions != labels).nonzero(as_tuple=True)[0]
    diff_teacher_logits = teacher_logits[diff_indices]
    diff_student_logits = students_logits[diff_indices]
    #print(diff_student_logits)
    
    return kl_loss(F.log_softmax(diff_student_logits/T,dim=1),F.softmax(diff_teacher_logits.detach(),dim=1))
def pull_loss(teacher_logits, students_logits, labels,T=1):#train_batch_labels
    '''print(teacher_logits.shape)
    print(students_logits.shape)
    print(labels.shape)'''
    teacher_predictions = torch.argmax(teacher_logits, dim=1)
    #print(teacher_predictions.shape)
    diff_indices = (teacher_predictions == labels).nonzero(as_tuple=True)[0]
    diff_teacher_logits = teacher_logits[diff_indices]
    diff_student_logits = students_logits[diff_indices]
    #print(diff_student_logits)
    return kl_loss(F.log_softmax(diff_student_logits/T,dim=1),F.softmax(diff_teacher_logits.detach(),dim=1))

teacher = wideresnet()#WideResNet()
teacher1_path =  'models/model_cifar_wrn.pt'
#state_dict = torch.load(teacher1_path)
#teacher.load_state_dict(state_dict)

state_dict = torch.load(teacher1_path,map_location=torch.device('cpu'))#["model"]
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
teacher.load_state_dict(new_state_dict)

#teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
# teacher = teacher.half()
#teacher.eval()
teacher_lr = 0.0001
ADV_teacher_optimizer = optim.SGD(teacher.parameters(), lr=teacher_lr, momentum=0.1, weight_decay=2e-4)
ADV_teacher_loss_CE = torch.nn.CrossEntropyLoss().cuda()
teacher.train()


teacher_nat = cifar10_resnet56()#resnet56()
teacher2_path = 'models/nat_teacher_checkpoint/cifar10_resnnet56.pth'
#state_dict_1 = torch.load(teacher2_path)
#teacher_nat.load_state_dict(state_dict_1)

state_dict = torch.load(teacher2_path,map_location=torch.device('cpu'))
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
teacher_nat.load_state_dict(new_state_dict)

#teacher = torch.nn.DataParallel(teacher)
teacher_nat = teacher_nat.cuda()
teacher_nat.eval()


weight_learn_rate = 0.025
temp_learn_rate = 0.001

ce_loss = torch.nn.CrossEntropyLoss().cuda()
ce_loss_test = torch.nn.CrossEntropyLoss(reduction='none')
best_accuracy = 0

temp_adv = 1
temp_nat = 1

temp_max = 10
temp_min = 1

logger.info('''
CIARD
push label T=5
Lr stage decay
push_loss(nat_adv_logits,student_adv_logits,train_batch_labels) 
teacher lr weight decay from 0.0001 to 0 with smooth decay
epoch = 300 coslr
''')

for epoch in range(begin_epoch,epochs+1):
    logger.info('the {}th epoch '.format(epoch)) 
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader): 
        student.train()
        teacher.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        ADV_teacher_optimizer.zero_grad()
         
        student.train()
        student_nat_logits = student(train_batch_data)
        with torch.no_grad():
            teacher_nat_logits = teacher_nat(train_batch_data)
            adv_teacher_nat = teacher(train_batch_data)

        student_adv_logits,teacher_adv_logits,nat_adv_logits = robust_inner_loss_push(student,teacher,teacher_nat,
                                                                                        train_batch_data,train_batch_labels,
                                                                                        optimizer,ADV_teacher_optimizer,
                                                                                        step_size=2/255.0,
                                                                                        epsilon=epsilon,perturb_steps=10)
    
        kl_Loss1 = kl_loss(F.log_softmax(student_adv_logits,dim=1),F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1))
        kl_Loss2 = kl_loss(F.log_softmax(student_nat_logits,dim=1),F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1))
        kl_Loss1 = torch.mean(kl_Loss1)
        kl_Loss2 = torch.mean(kl_Loss2)
        adv_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1)))
        nat_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1)))
        temp_adv = temp_adv - temp_learn_rate * torch.sign((adv_teacher_entropy.detach() / nat_teacher_entropy.detach() - 1)).item()
        temp_nat = temp_nat - temp_learn_rate * torch.sign((nat_teacher_entropy.detach() / adv_teacher_entropy.detach() - 1)).item()
        temp_adv = max(min(temp_max, temp_adv), temp_min)
        temp_nat = max(min(temp_max, temp_nat), temp_min)
        if init_loss_nat == None:
            init_loss_nat = kl_Loss2.item()
        if init_loss_adv == None:
            init_loss_adv = kl_Loss1.item()
        G_avg = (kl_Loss1.item() + kl_Loss2.item()) / len(weight)
        lhat_adv = kl_Loss1.item() / init_loss_adv
        lhat_nat = kl_Loss2.item() / init_loss_nat
        lhat_avg = (lhat_adv + lhat_nat) / len(weight)
        inv_rate_adv = lhat_adv / lhat_avg
        inv_rate_nat = lhat_nat / lhat_avg
        weight["nat_loss"] = weight["nat_loss"] - weight_learn_rate *(weight["nat_loss"] - inv_rate_nat/(inv_rate_adv + inv_rate_nat))
        weight["adv_loss"] = weight["adv_loss"] - weight_learn_rate *(weight["adv_loss"] - inv_rate_adv/(inv_rate_adv + inv_rate_nat))
        num_losses = len(weight)
        if weight["adv_loss"] <0:
            weight["adv_loss"] = 0
        if weight["nat_loss"]< 0:
            weight["nat_loss"] = 0
        coef = 1.0/(weight["adv_loss"] + weight["nat_loss"])
        weight["adv_loss"] *= coef
        weight["nat_loss"] *= coef
        total_loss = weight["adv_loss"]*kl_Loss1 + weight["nat_loss"]*kl_Loss2


        kl_Loss3 = push_loss(nat_adv_logits,student_adv_logits,train_batch_labels) 
        if(torch.isnan(kl_Loss3).any() or kl_Loss3.numel() == 0):
            kl_Loss3 = torch.tensor(0.0)
        else:
            kl_Loss3 = torch.mean(kl_Loss3)
            loss3_weight = scale_to_magnitude(float(kl_Loss1.item()), float(kl_Loss2.item()), float(kl_Loss3.item())) 
            total_loss -= loss3_weight*kl_Loss3
        '''
        kl_Loss4 = push_loss(adv_teacher_nat,student_nat_logits,train_batch_labels) 
        if(torch.isnan(kl_Loss4).any() or kl_Loss4.numel() == 0):
            kl_Loss4 = torch.tensor(0.0)
        else:
            kl_Loss4 = torch.mean(kl_Loss4)
            loss4_weight = scale_to_magnitude(float(kl_Loss1.item()), float(kl_Loss2.item()), float(kl_Loss4.item()))
            total_loss -= loss4_weight*kl_Loss4
        '''

        if epoch < 150:
            lr = 0.1
        else:
            cosine_term = 0.5 + 0.5 * np.cos(np.pi * (epoch - 150) / (300 - 150))
            exponential_decay = np.exp(-0.01 * (epoch - 150) ** 2 / (300 - 150) ** 2)
            lr = 0.1 * cosine_term * exponential_decay

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if epoch < 50:
            teacher_lr = 0
        else:
            base_lr = 0.0001
            min_lr = 0
            cosine_term = 0.5 + 0.5 * np.cos(np.pi * (epoch - 50) / (300 - 50))
            exponential_decay = np.exp(-0.01 * (epoch - 50) ** 2 / (300 - 50) ** 2)
            teacher_lr = min_lr + (base_lr - min_lr) * cosine_term*exponential_decay
            
        for param_group in ADV_teacher_optimizer.param_groups:
            param_group['lr'] = teacher_lr
        if epoch in [215,260,285]:
            weight_learn_rate *= 0.1
            temp_learn_rate *= 0.1
                    
        student.train()
        total_loss.backward()
        optimizer.step()
        ADV_teacher_loss = ADV_teacher_loss_CE(teacher_adv_logits,train_batch_labels)
        if(epoch>50):
            ADV_teacher_loss.backward()
            ADV_teacher_optimizer.step()
        if step%100 == 0:
            text = 'lr:' + str(lr) 
            text += ' weight_nat: {}, nat_loss: {}, weight_adv: {}, adv_loss: {}'.format(weight["nat_loss"], kl_Loss2.item(), weight["adv_loss"], kl_Loss1.item()) 
            text += " weight-klloss3 " + str(loss3_weight) + " Loss3: " + str(kl_Loss3.item()) 
            logger.info(text) 
        

    if epoch == 1 or epoch%10==  0 or epoch >= 250: 
        loss_nat_test = AverageMeter()
        loss_adv_test = AverageMeter()

        student.eval()
        teacher.eval()
        teacher_nat.eval()

        optimizer.zero_grad()
        ADV_teacher_optimizer.zero_grad()
        test_accs = []
        test_accs_naturals = []
        teacher_test_accs = []
        teacher_test_accs_naturals = []

        nat_teacher_test_accs = []
        nat_teacher_test_accs_naturals = []


        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            with torch.no_grad():
                logits = student(test_ifgsm_data)
                loss = ce_loss(logits, test_batch_labels)
            loss = loss.float()
            loss_adv_test.update(loss.item(), test_batch_data.size(0))
            
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
            teacher_logits = teacher(test_ifgsm_data)
            teacher_predictions = np.argmax(teacher_logits.cpu().detach().numpy(),axis=1)
            teacher_predictions = teacher_predictions - test_batch_labels.cpu().detach().numpy()
            teacher_test_accs = teacher_test_accs + teacher_predictions.tolist()

            nat_teacher_logits = teacher_nat(test_ifgsm_data)
            nat_teacher_predictions = np.argmax(nat_teacher_logits.cpu().detach().numpy(),axis=1)
            nat_teacher_predictions = nat_teacher_predictions - test_batch_labels.cpu().detach().numpy()
            nat_teacher_test_accs = nat_teacher_test_accs + nat_teacher_predictions.tolist()


        test_accs = np.array(test_accs)
        test_adv = np.sum(test_accs==0)/len(test_accs)
        teacher_test_accs = np.array(teacher_test_accs)
        teacher_test_acc = np.sum(teacher_test_accs==0)/len(teacher_test_accs)

        nat_teacher_test_accs = np.array(nat_teacher_test_accs)
        nat_teacher_test_acc = np.sum(nat_teacher_test_accs==0)/len(nat_teacher_test_accs)
        
        text = f'student robust acc {np.sum(test_accs==0)/len(test_accs):.4f}, teacher robust acc {np.sum(teacher_test_accs==0)/len(teacher_test_accs):.4f}, nat teacher robust acc {np.sum(nat_teacher_test_accs==0)/len(nat_teacher_test_accs):.4f}'
        logger.info(text)

        for step,(test_batch_data,test_batch_labels) in enumerate(testloader): 
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            with torch.no_grad():
                logits = student(test_batch_data)
                loss = ce_loss(logits, test_batch_labels)
            loss = loss.float()
            loss_nat_test.update(loss.item(), test_batch_data.size(0))
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs_naturals = test_accs_naturals + predictions.tolist()

            teacher_logits = teacher(test_batch_data)
            teacher_predictions = np.argmax(teacher_logits.cpu().detach().numpy(),axis=1)
            teacher_predictions = teacher_predictions - test_batch_labels.cpu().detach().numpy()
            teacher_test_accs_naturals = teacher_test_accs_naturals + teacher_predictions.tolist()

            nat_teacher_logits = teacher_nat(test_batch_data)
            nat_teacher_predictions = np.argmax(nat_teacher_logits.cpu().detach().numpy(),axis=1)
            nat_teacher_predictions = nat_teacher_predictions - test_batch_labels.cpu().detach().numpy()
            nat_teacher_test_accs_naturals = nat_teacher_test_accs_naturals + nat_teacher_predictions.tolist()
        test_accs_naturals = np.array(test_accs_naturals)
        test_nat = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
        teacher_test_accs_naturals = np.array(teacher_test_accs_naturals)
        teacher_test_accs_natural = np.sum(teacher_test_accs_naturals==0)/len(teacher_test_accs_naturals)

        nat_teacher_test_accs_naturals = np.array(nat_teacher_test_accs_naturals)
        nat_teacher_test_accs_natural = np.sum(nat_teacher_test_accs_naturals==0)/len(nat_teacher_test_accs_naturals)

        if epoch%50 == 0 :
            state = { 'model': student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/' + prefix + "/student_" + str(epoch)+ '.pth')
            state = { 'model': teacher.state_dict(),
                'optimizer': ADV_teacher_optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/'+ prefix + "/teacher_" + str(epoch)+ '.pth')
        if epoch > 250:
            state = { 'model': student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/' + prefix + "/student_latest.pth")
            state = { 'model': teacher.state_dict(),
                'optimizer': ADV_teacher_optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/'+ prefix + "/teacher_latest.pth")
        if (test_nat + test_adv) / 2 > best_accuracy:
            best_accuracy = (test_nat + test_adv)/2
            state = { 'model': student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/' + prefix + "/student_best"+ '.pth')
            state = { 'model': teacher.state_dict(),
                'optimizer': ADV_teacher_optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,'./model/' + prefix + "/teacher_best"+ '.pth')
            logger.info("best accuracy:"+str(best_accuracy))
            
        text = f'student natural acc {np.sum(test_accs_naturals==0)/len(test_accs_naturals):.4f}, adv teacher natural acc {np.sum(teacher_test_accs_naturals==0)/len(teacher_test_accs_naturals):.4f}, nat teacher natural acc {np.sum(nat_teacher_test_accs_naturals==0)/len(nat_teacher_test_accs_naturals):.4f}'
        logger.info(text)
        
        test_acc = np.sum(test_accs==0)/len(test_accs)
        test_accs_natural = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
        with open('./model/' + prefix+ '/'+ draw_file,'a') as f:
            text = str(epoch) + " " + str(test_acc) + " " + str(test_accs_natural) + " " + str(teacher_test_acc) + " "+ str(teacher_test_accs_natural)+ " "+ str(nat_teacher_test_acc) + " "+ str(nat_teacher_test_accs_natural)+'\n'
            f.write(text)