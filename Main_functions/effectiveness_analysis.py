# !/usr/bin/env python
# coding: utf-8


## structure 
from tabulate import tabulate
table = [['Model','structure','F1','Exact_match','Word_match','String_match','Levenshtein similarity']]

structure_list = ['full', 'remove_norm', 'remove_fs', 'remove_svd', 'mean_only', 'max_only']
for stru_i in range(len(structure_list)):
    stru = structure_list[stru_i]
    
    total_num = 0
    ori_f1, ori_ex, ori_wm, ori_sm, ori_LS = 0,0,0,0,0
    later_f1, later_ex, later_wm, later_sm, later_LS = 0,0,0,0,0
    
    for b in ['6','7','8','9']:
        new_BioASQ_xB_SQuAD=Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',
                                                        allow_pickle='TRUE').item())
        raw_pred_BioBERTlarge_newBioASQ_xB=np.load('./BioASQ/Task'+b+'BGoldenEnriched/raw_pred_BioBERTlarge_newBioASQ_'+b+'B.npy'
                                                ,allow_pickle ='TRUE') 
        #save: 
        #cID_with_fID_BioASQ_xB_BioBERT = mapping_cID_with_fID(new_BioASQ_xB_SQuAD, tokenizer = tokenizer_biobert_large)
        #np.save("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy", cID_with_fID_BioASQ_xB_BioBERT) 
        #load: 
        cID_with_fID_BioASQ_xB_BioBERT = np.load("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy",
                                         allow_pickle=True)

        cID_with_fID_BioASQ_xB_BioBERT_dic = dict(cID_with_fID_BioASQ_xB_BioBERT.item())
        cID_with_fID_BioASQ_xB_BioBERT_dic2 = {v[0]: v for k, v in cID_with_fID_BioASQ_xB_BioBERT_dic.items()}

        #Create dictionary to store question index, number of augumentation, number of splits
        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        Q_index_aug_split_dic = {} #record ori question index, number of aug q, and number of split
        allFeature_sofar = 0  #number of features so far 

        b_num = len(indexNum_dic)
        total_num += b_num
        for q_i, aug in indexNum_dic.items(): #for each original question, its aug number
            split = len(cID_with_fID_BioASQ_xB_BioBERT_dic2[allFeature_sofar]) #how many splits for this q(note all aug qs have same length)
            Q_index_aug_split_dic[q_i] = {'aug':aug, 'split':split}
            allFeature_sofar += aug * split

        ori_scores, MSQ_scores, matrix_for_plot = multi_score_2_single_score(new_BioASQ_xB_SQuAD, raw_pred_BioBERTlarge_newBioASQ_xB, Q_index_aug_split_dic, 
                                                                                       rank = 1, augQ_fs0 = 0.01, LS_threshold = 0.85, structure = stru)

        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        rangeIndex = np.cumsum([0]+list(indexNum_dic.values()))[:-1]
        print('------------------BioASQ_',b,'B SQuAD------------------')
        ori_result, ori_p, ori_r = calculate_ex_f1(ori_scores[0], ori_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)
        MSQ_result, msq_p, msq_r = calculate_ex_f1(MSQ_scores[0], MSQ_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)

        ori_pred_text = [i['prediction_text'] for i in ori_p] 
        msq_pred_text = [i['prediction_text'] for i in msq_p]  
        groundtruth_text = [i['answers']['text'] for i in ori_r] 

        word_matching_ori = word_matching_scores(ori_pred_text ,groundtruth_text)
        word_matching_msq = word_matching_scores(msq_pred_text ,groundtruth_text)
        str_matching_ori = string_matching_scores(ori_pred_text ,groundtruth_text)
        str_matching_msq = string_matching_scores(msq_pred_text ,groundtruth_text)
        LS_ori = Levenshtein_similarity(ori_pred_text,groundtruth_text)
        LS_msq = Levenshtein_similarity(msq_pred_text,groundtruth_text)

        ori_f1 += ori_result['f1'] * b_num 
        ori_ex += ori_result['exact_match'] * b_num 
        ori_wm += word_matching_ori * b_num 
        ori_sm += str_matching_ori * b_num 
        ori_LS += LS_ori * b_num 

        later_f1 += MSQ_result['f1'] * b_num 
        later_ex += MSQ_result['exact_match'] * b_num 
        later_wm += word_matching_msq * b_num 
        later_sm += str_matching_msq * b_num 
        later_LS += LS_msq * b_num 


    if stru_i == 0:
        table.append(['Baseline BioBERT', '-', str(round(ori_f1/total_num,2))+"%",  
                                            str(round(ori_ex/total_num,2))+"%",
                                            str(round(ori_wm/total_num,2))+"%",
                                            str(round(ori_sm/total_num,2))+"%",
                                            str(round(ori_LS/total_num,2))+"%"])
        table.append(['MSQ_BioBERT', stru , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
    else: 
        table.append(['MSQ_BioBERT', stru , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
        
print(tabulate(table))





## Levenshtein similarity Threshold (LST) 
from tabulate import tabulate
table = [['Model','LS_threshold','F1','Exact_match','Word_match','String_match','Levenshtein similarity']]

LS_list = [0, 0.1, 0.3, 0.5, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0] 
for lsi in range(len(LS_list)):
    LS = LS_list[lsi]
    
    total_num = 0
    ori_f1, ori_ex, ori_wm, ori_sm, ori_LS = 0,0,0,0,0
    later_f1, later_ex, later_wm, later_sm, later_LS = 0,0,0,0,0
    
    for b in ['6','7','8','9']:
        new_BioASQ_xB_SQuAD=Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',
                                                        allow_pickle='TRUE').item())
        raw_pred_BioBERTlarge_newBioASQ_xB=np.load('./BioASQ/Task'+b+'BGoldenEnriched/raw_pred_BioBERTlarge_newBioASQ_'+b+'B.npy'
                                                ,allow_pickle ='TRUE') 
        #save: 
        #cID_with_fID_BioASQ_xB_BioBERT = mapping_cID_with_fID(new_BioASQ_xB_SQuAD, tokenizer = tokenizer_biobert_large)
        #np.save("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy", cID_with_fID_BioASQ_xB_BioBERT) 
        #load: 
        cID_with_fID_BioASQ_xB_BioBERT = np.load("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy",
                                         allow_pickle=True)

        cID_with_fID_BioASQ_xB_BioBERT_dic = dict(cID_with_fID_BioASQ_xB_BioBERT.item())
        cID_with_fID_BioASQ_xB_BioBERT_dic2 = {v[0]: v for k, v in cID_with_fID_BioASQ_xB_BioBERT_dic.items()}

        #Create dictionary to store question index, number of augumentation, number of splits
        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        Q_index_aug_split_dic = {} #record ori question index, number of aug q, and number of split
        allFeature_sofar = 0  #number of features so far 

        b_num = len(indexNum_dic)
        total_num += b_num
        for q_i, aug in indexNum_dic.items(): #for each original question, its aug number
            split = len(cID_with_fID_BioASQ_xB_BioBERT_dic2[allFeature_sofar]) #how many splits for this q(note all aug qs have same length)
            Q_index_aug_split_dic[q_i] = {'aug':aug, 'split':split}
            allFeature_sofar += aug * split

        ori_scores, MSQ_scores, matrix_for_plot = multi_score_2_single_score(new_BioASQ_xB_SQuAD, raw_pred_BioBERTlarge_newBioASQ_xB, Q_index_aug_split_dic, 
                                                                                       rank = 1, augQ_fs0 = 0.01, LS_threshold = LS, structure = 'full')

        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        rangeIndex = np.cumsum([0]+list(indexNum_dic.values()))[:-1]
        print('------------------BioASQ_',b,'B SQuAD------------------')
        ori_result, ori_p, ori_r = calculate_ex_f1(ori_scores[0], ori_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)
        MSQ_result, msq_p, msq_r = calculate_ex_f1(MSQ_scores[0], MSQ_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)

        ori_pred_text = [i['prediction_text'] for i in ori_p] 
        msq_pred_text = [i['prediction_text'] for i in msq_p]  
        groundtruth_text = [i['answers']['text'] for i in ori_r] 

        word_matching_ori = word_matching_scores(ori_pred_text ,groundtruth_text)
        word_matching_msq = word_matching_scores(msq_pred_text ,groundtruth_text)
        str_matching_ori = string_matching_scores(ori_pred_text ,groundtruth_text)
        str_matching_msq = string_matching_scores(msq_pred_text ,groundtruth_text)
        LS_ori = Levenshtein_similarity(ori_pred_text,groundtruth_text)
        LS_msq = Levenshtein_similarity(msq_pred_text,groundtruth_text)

        ori_f1 += ori_result['f1'] * b_num 
        ori_ex += ori_result['exact_match'] * b_num 
        ori_wm += word_matching_ori * b_num 
        ori_sm += str_matching_ori * b_num 
        ori_LS += LS_ori * b_num 

        later_f1 += MSQ_result['f1'] * b_num 
        later_ex += MSQ_result['exact_match'] * b_num 
        later_wm += word_matching_msq * b_num 
        later_sm += str_matching_msq * b_num 
        later_LS += LS_msq * b_num 


    if lsi == 0:
        table.append(['Baseline BioBERT', '-', str(round(ori_f1/total_num,2))+"%",  
                                            str(round(ori_ex/total_num,2))+"%",
                                            str(round(ori_wm/total_num,2))+"%",
                                            str(round(ori_sm/total_num,2))+"%",
                                            str(round(ori_LS/total_num,2))+"%"])
        table.append(['MSQ_BioBERT', LS , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
    else: 
        table.append(['MSQ_BioBERT', LS , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
        
print(tabulate(table))







## Initial word-frequency scores (IWFS)
from tabulate import tabulate
table = [['Model','Initial_FS','F1','Exact_match','Word_match','String_match','Levenshtein similarity']]

augQ_fs0_list = [0.1, 0.05, 0.01, 0.005, 0.0001] 
for fsi in range(len(augQ_fs0_list)):
    FS = augQ_fs0_list[fsi]
    
    total_num = 0
    ori_f1, ori_ex, ori_wm, ori_sm, ori_LS = 0,0,0,0,0
    later_f1, later_ex, later_wm, later_sm, later_LS = 0,0,0,0,0
    
    for b in ['6','7','8','9']:
        new_BioASQ_xB_SQuAD=Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',
                                                        allow_pickle='TRUE').item())
        raw_pred_BioBERTlarge_newBioASQ_xB=np.load('./BioASQ/Task'+b+'BGoldenEnriched/raw_pred_BioBERTlarge_newBioASQ_'+b+'B.npy'
                                                ,allow_pickle ='TRUE') 
    	#save: 
        #cID_with_fID_BioASQ_xB_BioBERT = mapping_cID_with_fID(new_BioASQ_xB_SQuAD, tokenizer = tokenizer_biobert_large)
        #np.save("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy", cID_with_fID_BioASQ_xB_BioBERT) 
        #load: 
        cID_with_fID_BioASQ_xB_BioBERT = np.load("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy",
                                         allow_pickle=True)

        cID_with_fID_BioASQ_xB_BioBERT_dic = dict(cID_with_fID_BioASQ_xB_BioBERT.item())
        cID_with_fID_BioASQ_xB_BioBERT_dic2 = {v[0]: v for k, v in cID_with_fID_BioASQ_xB_BioBERT_dic.items()}

        #Create dictionary to store question index, number of augumentation, number of splits
        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        Q_index_aug_split_dic = {} #record ori question index, number of aug q, and number of split
        allFeature_sofar = 0  #number of features so far 

        b_num = len(indexNum_dic)
        total_num += b_num
        for q_i, aug in indexNum_dic.items(): #for each original question, its aug number
            split = len(cID_with_fID_BioASQ_xB_BioBERT_dic2[allFeature_sofar]) #how many splits for this q(note all aug qs have same length)
            Q_index_aug_split_dic[q_i] = {'aug':aug, 'split':split}
            allFeature_sofar += aug * split

        ori_scores, MSQ_scores, matrix_for_plot = multi_score_2_single_score(new_BioASQ_xB_SQuAD, raw_pred_BioBERTlarge_newBioASQ_xB, Q_index_aug_split_dic, 
                                                                                       rank = 1, augQ_fs0 = FS, LS_threshold = 0.85, structure = 'full')

        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        rangeIndex = np.cumsum([0]+list(indexNum_dic.values()))[:-1]
        print('------------------BioASQ_',b,'B SQuAD------------------')
        ori_result, ori_p, ori_r = calculate_ex_f1(ori_scores[0], ori_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)
        MSQ_result, msq_p, msq_r = calculate_ex_f1(MSQ_scores[0], MSQ_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)

        ori_pred_text = [i['prediction_text'] for i in ori_p] 
        msq_pred_text = [i['prediction_text'] for i in msq_p]  
        groundtruth_text = [i['answers']['text'] for i in ori_r] 

        word_matching_ori = word_matching_scores(ori_pred_text ,groundtruth_text)
        word_matching_msq = word_matching_scores(msq_pred_text ,groundtruth_text)
        str_matching_ori = string_matching_scores(ori_pred_text ,groundtruth_text)
        str_matching_msq = string_matching_scores(msq_pred_text ,groundtruth_text)
        LS_ori = Levenshtein_similarity(ori_pred_text,groundtruth_text)
        LS_msq = Levenshtein_similarity(msq_pred_text,groundtruth_text)

        ori_f1 += ori_result['f1'] * b_num 
        ori_ex += ori_result['exact_match'] * b_num 
        ori_wm += word_matching_ori * b_num 
        ori_sm += str_matching_ori * b_num 
        ori_LS += LS_ori * b_num 

        later_f1 += MSQ_result['f1'] * b_num 
        later_ex += MSQ_result['exact_match'] * b_num 
        later_wm += word_matching_msq * b_num 
        later_sm += str_matching_msq * b_num 
        later_LS += LS_msq * b_num 


    if fsi == 0:
        table.append(['Baseline BioBERT', '-', str(round(ori_f1/total_num,2))+"%",  
                                            str(round(ori_ex/total_num,2))+"%",
                                            str(round(ori_wm/total_num,2))+"%",
                                            str(round(ori_sm/total_num,2))+"%",
                                            str(round(ori_LS/total_num,2))+"%"])
        table.append(['MSQ_BioBERT', FS , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
    else: 
        table.append(['MSQ_BioBERT', FS , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
        
print(tabulate(table))





## rank
from tabulate import tabulate
table = [['Model','SVD_rank','F1','Exact_match','Word_match','String_match','Levenshtein similarity']]

rank_list = [10,4,3,2,1]
for ri in range(len(rank_list)):
    R = rank_list[ri]
    
    total_num = 0
    ori_f1, ori_ex, ori_wm, ori_sm, ori_LS = 0,0,0,0,0
    later_f1, later_ex, later_wm, later_sm, later_LS = 0,0,0,0,0
    
    for b in ['6','7','8','9']:
        new_BioASQ_xB_SQuAD=Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',
                                                        allow_pickle='TRUE').item())
        raw_pred_BioBERTlarge_newBioASQ_xB=np.load('./BioASQ/Task'+b+'BGoldenEnriched/raw_pred_BioBERTlarge_newBioASQ_'+b+'B.npy'
                                                ,allow_pickle ='TRUE') 
        #load: 
        cID_with_fID_BioASQ_xB_BioBERT = np.load("./BioASQ/Task"+b+"BGoldenEnriched/cID_with_fID_BioASQ_"+b+"B_BioBERT.npy",
                                         allow_pickle=True)

        cID_with_fID_BioASQ_xB_BioBERT_dic = dict(cID_with_fID_BioASQ_xB_BioBERT.item())
        cID_with_fID_BioASQ_xB_BioBERT_dic2 = {v[0]: v for k, v in cID_with_fID_BioASQ_xB_BioBERT_dic.items()}

        #Create dictionary to store question index, number of augumentation, number of splits
        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        Q_index_aug_split_dic = {} #record ori question index, number of aug q, and number of split
        allFeature_sofar = 0  #number of features so far 

        b_num = len(indexNum_dic)
        total_num += b_num
        for q_i, aug in indexNum_dic.items(): #for each original question, its aug number
            split = len(cID_with_fID_BioASQ_xB_BioBERT_dic2[allFeature_sofar]) #how many splits for this q(note all aug qs have same length)
            Q_index_aug_split_dic[q_i] = {'aug':aug, 'split':split}
            allFeature_sofar += aug * split

        ori_scores, MSQ_scores, matrix_for_plot = multi_score_2_single_score(new_BioASQ_xB_SQuAD, raw_pred_BioBERTlarge_newBioASQ_xB, Q_index_aug_split_dic, 
                                                                                       rank = R, augQ_fs0 = 0.01, LS_threshold = 0.85, structure = 'full')

        indexNum_dic = Counter([i.split('_')[0] for i in new_BioASQ_xB_SQuAD['new_id']]) 
        rangeIndex = np.cumsum([0]+list(indexNum_dic.values()))[:-1]
        print('------------------BioASQ_',b,'B SQuAD------------------')
        ori_result, ori_p, ori_r = calculate_ex_f1(ori_scores[0], ori_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)
        MSQ_result, msq_p, msq_r = calculate_ex_f1(MSQ_scores[0], MSQ_scores[1], Dataset.from_dict(new_BioASQ_xB_SQuAD[rangeIndex]), tokenizer_biobert_large)

        ori_pred_text = [i['prediction_text'] for i in ori_p] 
        msq_pred_text = [i['prediction_text'] for i in msq_p]  
        groundtruth_text = [i['answers']['text'] for i in ori_r] 

        word_matching_ori = word_matching_scores(ori_pred_text ,groundtruth_text)
        word_matching_msq = word_matching_scores(msq_pred_text ,groundtruth_text)
        str_matching_ori = string_matching_scores(ori_pred_text ,groundtruth_text)
        str_matching_msq = string_matching_scores(msq_pred_text ,groundtruth_text)
        LS_ori = Levenshtein_similarity(ori_pred_text,groundtruth_text)
        LS_msq = Levenshtein_similarity(msq_pred_text,groundtruth_text)

        ori_f1 += ori_result['f1'] * b_num 
        ori_ex += ori_result['exact_match'] * b_num 
        ori_wm += word_matching_ori * b_num 
        ori_sm += str_matching_ori * b_num 
        ori_LS += LS_ori * b_num 

        later_f1 += MSQ_result['f1'] * b_num 
        later_ex += MSQ_result['exact_match'] * b_num 
        later_wm += word_matching_msq * b_num 
        later_sm += str_matching_msq * b_num 
        later_LS += LS_msq * b_num 


    if ri == 0:
        table.append(['Baseline BioBERT', '-', str(round(ori_f1/total_num,2))+"%",  
                                            str(round(ori_ex/total_num,2))+"%",
                                            str(round(ori_wm/total_num,2))+"%",
                                            str(round(ori_sm/total_num,2))+"%",
                                            str(round(ori_LS/total_num,2))+"%"])
        table.append(['MSQ_BioBERT', R , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
    else: 
        table.append(['MSQ_BioBERT', R , str(round(later_f1/total_num,2))+"%", 
                                            str(round(later_ex/total_num,2))+"%",
                                            str(round(later_wm/total_num,2))+"%",
                                            str(round(later_sm/total_num,2))+"%",
                                            str(round(later_LS/total_num,2))+"%"])
        
print(tabulate(table))






