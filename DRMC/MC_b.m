% MC-b by Miao Fan 
% Tsinghua University
% fanmiao.cslt.thu@gmail.com

function Out = MC_b(Xtrain, Ytrain, Xtest, Ytest)
    
    Out.r = [];
    Out.prec = [];
    Out.rec = [];
    
    [numOfXtrain, demXtrain] = size(Xtrain);
    [numOfYtrain, demYtrain] = size(Ytrain);

    B = zeros(1, demYtrain);

    X = [Xtrain; Xtest];
    [x, y] = find(X > 0);
    OmigaX = [x, y];
    [r, c] = size(OmigaX);
    numOfOmigaX = r;

    Y = [Ytrain; zeros(size(Ytest))];
    [x, y] = find(Y > 0);
    OmigaY = [x, y];
    [r, c] = size(OmigaY);
    numOfOmigaY = r;
    
    Z = [X, Y];
    params = getParams(Z, numOfOmigaY, numOfOmigaX);
    
    mu = params.mus;
    muf = params.muf;
    
    for i = 1 : params.maxOuterItr

        for j = 1 : params.maxInnerItr
            Zp = Z;
            gb = getGB(params.lamda, numOfOmigaY, OmigaY, X, Y, Z, B);
            B = B - params.taub * gb;
            gz = getGZ(params.lamda, numOfOmigaY, numOfOmigaX, OmigaY, OmigaX, Y, X, Z, B);
            A = Z - params.tauz * gz;

            [U, S, V] = svd(A);

            S = max(0,S-params.tauz * mu);
            Z = U * S * V';
            
            ra = rank(Z)

%           prec = getPrec(Z, Ytest);
%             
%             if(size(Out.r) ~= 0 & Out.r(end) == ra)
%                 
%                Out.prec(end) = max(Out.prec(end), prec); 
%             else
%                 Out.r = [Out.r; ra];             
%                 Out.prec = [Out.prec; prec];           
%             end
%             
% %             rankList = Out.r
% %             precisionList = Out.prec
% 
%             Out.r
%             evaluation(Z, Ytest, 0);

            Out.r = [Out.r; ra];
            result = evaluation(Z, Ytest, 0);
            Out.prec = [Out.prec; result.preList];
            Out.rec = [Out.rec; result.recList];
            
            if(ra <= params.rank_b)
                Out.Z = Z;
                return;
            end
            
            if (norm(Zp-Z, 'fro') / max(1.0, norm(Zp, 'fro')) <= params.tol)
                if(mu == muf)
                    Out.Z = Z;
                    return;
                else
                    break;
                end
            end
        end
        mu = max(mu * params.eta, muf);
    end  
end

    %
    % inner gradient function for Z (MC_b)
    %

function gz = getGZ(lamda, numOfOmigaY, numOfOmigaX, OmigaY, OmigaX, Y, X, Z, B)

    [rowZ, columnZ] = size(Z);
    [rowX, columnX] = size(X);

    gz = zeros(rowZ, columnZ);

    for k = 1 : numOfOmigaY
        i = OmigaY(k, 1);
        j = OmigaY(k, 2);
        gz(i, j + columnX) = lamda / numOfOmigaY * (-Y(i, j) / (1 + exp(Y(i, j) * (Z(i, j + columnX) + B(1, j)))));        
    end

    for k = 1 : numOfOmigaX      
        i = OmigaX(k, 1);
        j = OmigaX(k, 2);
        gz(i, j) = 1.0 / numOfOmigaX * (- X(i, j) / (1 + exp(X(i, j) * Z(i, j))));        
    end
end

        
    %
    % inner gradient function for B
    %
    
function gb = getGB(lamda, numOfOmigaY, OmigaY, X, Y, Z, B)

    [rowB, columnB] = size(B);
    [rowY, columnY] = size(Y);
    [rowX, columnX] = size(X);
    
    gb = zeros(rowB, columnB);
    gy = zeros(rowY, columnY);
    
    for k = 1 : numOfOmigaY
        i = OmigaY(k, 1);
        j = OmigaY(k, 2);
        gy(i, j) = lamda / numOfOmigaY * (-Y(i, j)/ (1 + exp(Y(i, j) * (Z(i, j + columnX) + B(rowB, j)))));
    end
    
    for j = 1 : columnY
        sum = 0.0;
        for i = 1 : rowY
            sum = sum + gy(i, j);
        end
        gb(rowB, j) = sum;
    end   
end

%-------------------------
% Evaluation Function
%-------------------------

function prec = getPrec(Z1, Ytest)
    const = getConst();
    YPredict = Z1(const.ntrain + 1:const.ntrain + const.ntest, const.fd + 1:const.fd + const.ld);

    [m, n] = size(Ytest);
    count = 0;
    totalCount = 0;
    for i = 1 : m
        index1 = find(Ytest(i, :) > 0);
        [r, c] = size(index1);
        [value, index2] = sort(YPredict(i, :), 'descend');

        index2 = index2(1: c);
        [k, l] = size(intersect(index1, index2));
        count = count + l;
        totalCount = totalCount + c;

    end
    prec = count * 1.0 / totalCount;
end
    



