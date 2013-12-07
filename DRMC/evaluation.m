function result = evaluation(Z, Ytest, bias)
    const = getConst();
    [demZR, demZC] = size(Z);
    [demYR, demYC] = size(Ytest);
    
    YPredict = Z(demZR - demYR + 1 : demZR, const.fd + 1 + bias:const.fd + const.ld + bias);
    [E, I] = sort(YPredict(:), 'descend');
    [m, n] = size(Ytest);
    preList = [];
    recList = [];
    prebase = m * n;
    recbase = sum(Ytest(:) > 0);
    for i = 10 : 5 : prebase
%         YPredict(I(1:i))
        match = sum(Ytest(I(1:i)) > 0);
        
        precision = match * 1.0 / i;
        recall = match * 1.0 / recbase;
        preList = [preList, precision];
        recList = [recList, recall];
    end
    
    result.preList = preList; 
    result.recList = recList;

% % Compare Top100, Top500, Top100 - Rank
% 
%     const = getConst();
%     YPredict = Z(const.ntrain + 1:const.ntrain + const.ntest, const.fd + 1 + bias:const.fd + const.ld + bias);
%     [E, I] = sort(YPredict(:), 'descend');
%     % Top100
%     match100 = sum(Ytest(I(1:100)) > 0);
%     prebase100 = 100;
%     recbase100 = sum(Ytest(:) > 0);
%     
%     pre100 = match100 * 1.0 / prebase100;
%     rec100 = match100 * 1.0 / recbase100;
%     F100 = 2 * pre100 * rec100 / (pre100 + rec100)
%     
%     % Top500
%     match500 = sum(Ytest(I(1:500)) > 0);
%     prebase500 = 500;
%     recbase500 = sum(Ytest(:) > 0);
%     
%     pre500 = match500 * 1.0 / prebase500;
%     rec500 = match500 * 1.0 / recbase500;
%     F500 = 2 * pre500 * rec500 / (pre500 + rec500)
%     
%     % Top1000
%     match1000 = sum(Ytest(I(1:1000)) > 0);
%     prebase1000 = 1000;
%     recbase1000 = sum(Ytest(:) > 0);
%     
%     pre1000 = match1000 * 1.0 / prebase1000;
%     rec1000 = match1000 * 1.0 / recbase1000;
%     F1000 = 2 * pre1000 * rec1000 / (pre1000 + rec1000)

end