%--------------------
% Matrix Completion
%--------------------
%-----------------------------
clear
clc
Z0 = load('../Dataset/Feature Matrix/dataMatrix_NAACL_7313_3278_1947_51');
const = getConst();

Xtrain = Z0(1:const.ntrain, 1:const.fd);
Ytrain = Z0(1:const.ntrain, const.fd + 1 : const.fd + const.ld);
Xtest = Z0(const.ntrain + 1:const.ntest + const.ntrain, 1:const.fd);
Ytest = Z0(const.ntrain + 1:const.ntest + const.ntrain, const.fd + 1 : const.fd + const.ld);
Out_1 = MC_1(Xtrain, Ytrain, Xtest, Ytest);
% evaluation(Out_1.Z, Ytest, 1);
Out_b = MC_b(Xtrain, Ytrain, Xtest, Ytest);
% evaluation(Out_b.Z, Ytest, 0);
% 
% % plot(Out_1.r, Out_1.prec, '-ro', Out_b.r, Out_b.prec, '-g*', 'LineWidth', 2, 'MarkerSize', 3);
% % 
% % title('tiny-pos-matrix-completion');
% % xlabel('Rank');
% % ylabel('Precision');
% % legend('MC-1', 'MC-b');