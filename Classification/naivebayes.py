import numpy as np
# def naivebayes_classifier(all_condition_prob,dataset):
#     for data in dataset:
#         # assume that attributes are independant
#         attributes_condition_pro=1
#         for attribute in data:
#             pro=all_condition_prob
#             attributes_condition_pro*=pro
        
            
def conditional_prob(member_idx,attr_idx,total):
    condition_prob={}
    member_pro={}
    for k,v in member_idx.items():
        member_pro[k]=np.size(v)/total
    for attr_k in attr_idx.keys():
        for member_k in member_idx.keys():
            intersection=np.intersect1d(member_idx[member_k],attr_idx[attr_k])
            intersection_pro=np.size(intersection)/total
            condition_prob[member_k]= intersection_pro / member_pro[member_k]
            
    print(condition_prob)
    return condition_prob
if __name__ == "__main__":
    dataset=[]
    with open("training.txt","r") as file:
        for line in file:
            line=line.replace('{','')
            line=line.replace('}','')
            line=line.replace('\n','')
            sub_line=line.split(',')
            data=[None for _ in range(5)]
            for item in sub_line:
                token=item.split(' ')
                data[int(token[0])]=token[1]
            if(data[0]==None): data[0]="S"
            if(data[1]==None): data[1]=0
            if(data[2]==None): data[2]="Basic"
            dataset.append(data)
    dataset=np.array(dataset)
    dataset_cols=np.transpose(dataset)
    total=np.size(dataset[:,0])


    all_attr_idx=[]
    for index,data in enumerate(dataset_cols):
        unique_attributes=np.unique(data,return_counts=False)
        attr_idx={}
        for attr in unique_attributes:
            attr_idx[attr]=np.where(data==attr)
        all_attr_idx.append(attr_idx)



    all_condition_prob=[]
    member_idx=all_attr_idx[2]
    for idx,attribute in enumerate(all_attr_idx):
        if(idx==2): continue
        prob=conditional_prob(member_idx,attribute,total)
        all_condition_prob.append(prob)
    #naivebayes_classifier(all_condition_prob,dataset)

    


