import numpy as np
def accuracy(result,ground_truth):
    count=0
    for idx,member in enumerate(ground_truth):
        if(member==result[idx]):
            count+=1
    return count/np.size(ground_truth)
def naivebayes_classifier(all_condition_prob,dataset,member_pro):
    result=[]
    for data in dataset:
        # assume that attributes are independant
        # this line data's attributes to calculate the member probability
        attributes_condition_pro={'Basic': 1.0, 'Gold': 1.0, 'Normal': 1.0, 'Silver': 1.0}
        for idx,attribute in enumerate(data):

            ## calculate those attribute to members( basic, gold, normal, silver)
            for k,v in all_condition_prob[idx][attribute].items():
                attributes_condition_pro[k]*=v
        
        
        # calculate all conditional probability multiply member probability
        max=0
        max_key=""
        for k,v in attributes_condition_pro.items():
            pro=v*member_pro[k]
            if(pro>max): 
                max=pro
                max_key=k
        result.append(max_key)
    return np.array(result)

        
            
def conditional_prob(member_idx,attr_idx,total):
    condition_prob={}
    member_pro={}
    for k,v in member_idx.items():
        member_pro[k]=np.size(v)/total
    for attr_k in attr_idx.keys():
        condition_prob[attr_k]={}
        for member_k in member_idx.keys():
            intersection=np.intersect1d(member_idx[member_k],attr_idx[attr_k])
            intersection_pro=np.size(intersection)/total
            condition_prob[attr_k][member_k]= intersection_pro / member_pro[member_k]
            
    #print(condition_prob)
    return condition_prob,member_pro
def preprocessing(file_name):
    dataset=[]
    with open(file_name,"r") as file:
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
    return np.array(dataset)

if __name__ == "__main__":
    train_dataset=preprocessing("training.txt")
    dataset_cols=np.transpose(train_dataset)
    total=np.size(train_dataset[:,0])


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
        prob,member_pro=conditional_prob(member_idx,attribute,total)
        all_condition_prob.append(prob)


    ## all attribute condition probability based on member (attribute/member)
    #print(all_condition_prob)
    result=naivebayes_classifier(all_condition_prob,np.delete(train_dataset,2,axis=1),member_pro)

    acc=accuracy(result,dataset_cols[2])


    print(acc)