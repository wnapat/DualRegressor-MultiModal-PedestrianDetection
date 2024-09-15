function [miss,roc,misstable] = kaistEvalMulti( varargin )
% Evaluation detection results given ground truth.
%
% USAGE
%  [miss,roc,gt,dt] = kaistEval( pTest )
%
% INPUTS
%  pTest    - parameters (struct or name/value pairs)
%   .name     - ['REQ'] detection filename
%   .gtDir    - ['REQ'] dir containing test ground truth
%   .pLoad    - [] params for bbGt2>bbLoad for test data (see bbGt2>bbLoad)
%   .thr      - [.5] threshold on overlap area for comparing two bbs
%   .mul      - [0] if true allow multiple matches to each gt
%   .ref      - [10.^(-2:.25:0)] reference points (see bbGt2>compRoc)
%   .lims     - [3.1e-3 1e1 .05 1] plot axis limits
%   .show     - [0] optional figure number for display
%
% OUTPUTS
%  miss     - log-average miss rate computed at reference points
%  roc      - [nx3] n data points along roc of form [score fp tp]
%  gt       - [mx5] ground truth results [x y w h match] (see bbGt2>evalRes)
%  dt       - [nx6] detect results [x y w h score match] (see bbGt2>evalRes)
%
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]


% get parameters
dfs={ 'nameC','REQ', 'nameT','REQ', 'gtDir','REQ', 'threshold',.5, 'shift',0, 'pLoad',[], 'mul',0, ...
   'ref',10.^(-2:.25:0), 'lims',[3.1e-3 1e1 .05 1], 'subset', [], 'inputType','REQ', ...
   'show',0, 'clr', 'REQ' };
[nameC,nameT,gtDir,threshold,shift,pLoad,mul,ref,lims,subset,inputType,show,clr] = ...
  getPrmDflt(varargin,dfs,1);

% run evaluation using bbGt2
bbsNmC=[nameC '.txt'];
bbsNmT=[nameT '.txt'];

[gtC,dtC] = bbGt2('loadAll',gtDir,bbsNmC,pLoad,inputType{1},subset,0);
[gtT,dtT] = bbGt2('loadAll',gtDir,bbsNmT,pLoad,inputType{2},subset,shift);
[gtM,dtM] = evalResMulti(gtC,gtT,dtC,dtT,threshold,mul);
[fp,tp,score,allmiss] = bbGt2('compRoc',gtM,dtM,1,ref);
misstable = 1-allmiss;
miss=exp(mean(log(max(1e-10,1-allmiss))));
roc=[score fp tp];

if( ~show ), return; end
figure(show); 
plotRoc([fp tp],'logx',1,'logy',1,'xLbl','False positives per image',...
  'lims',lims,'color',clr,'smooth',1,'fpTarget',ref);

end

function [gtM,dtM] = evalResMulti( gt0C, gt0T, dt0C, dt0T, thr, mul )
% Evaluates detections against ground truth data.
%
% Uses modified Pascal criteria that allows for "ignore" regions. The
% Pascal criteria states that a ground truth bounding box (gtBb) and a
% detected bounding box (dtBb) match if their overlap area (oa):
%  oa(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(union(gtBb,dtBb))
% is over a sufficient threshold (typically .5). In the modified criteria,
% the dtBb can match any subregion of a gtBb set to "ignore". Choosing
% gtBb' in gtBb that most closely matches dtBb can be done by using
% gtBb'=intersect(dtBb,gtBb). Computing oa(gtBb',dtBb) is equivalent to
%  oa'(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(dtBb)
% For gtBb set to ignore the above formula for oa is used.
%
% Highest scoring detections are matched first. Matches to standard,
% (non-ignore) gtBb are preferred. Each dtBb and gtBb may be matched at
% most once, except for ignore-gtBb which can be matched multiple times.
% Unmatched dtBb are false-positives, unmatched gtBb are false-negatives.
% Each match between a dtBb and gtBb is a true-positive, except matches
% between dtBb and ignore-gtBb which do not affect the evaluation criteria.
%
% In addition to taking gt/dt results on a single image, evalRes() can take
% cell arrays of gt/dt bbs, in which case evaluation proceeds on each
% element. Use bbGt>loadAll() to load gt/dt for multiple images.
%
% Each gt/dt output row has a flag match that is either -1/0/1:
%  for gt: -1=ignore,  0=fn [unmatched],  1=tp [matched]
%  for dt: -1=ignore,  0=fp [unmatched],  1=tp [matched]
%
% USAGE
%  [gt, dt] = bbGt( 'evalRes', gt0, dt0, [thr], [mul] )
%
% INPUTS
%  gt0  - [mx5] ground truth array with rows [x y w h ignore]
%  dt0  - [nx5] detection results array with rows [x y w h score]
%  thr  - [.5] the threshold on oa for comparing two bbs
%  mul  - [0] if true allow multiple matches to each gt
%
% OUTPUTS
%  gt   - [mx5] ground truth results [x y w h match]
%  dt   - [nx6] detection results [x y w h score match]
%
% EXAMPLE
%
% See also bbGt, bbGt>compOas, bbGt>loadAll

% get parameters
if(nargin<5 || isempty(thr)), thr=.5; end
if(nargin<6 || isempty(mul)), mul=0; end

% assert(size(gt0C,1)==size(gt0T,1));
% assert(size(dt0C,1)==size(dt0T,1));

% if gt0 and dt0 are cell arrays run on each element in turn
if( iscell(gt0C) && iscell(dt0C) && iscell(gt0T) && iscell(dt0T)), n=length(gt0T);
  assert(length(dt0T)==n); gtC=cell(1,n); gtT=cell(1,n); dtC=gtC; dtT=gtT;
  assert(size(gt0C,2)==size(gt0T,2));
  assert(size(dt0C,2)==size(dt0T,2));
  assert(size(gt0C,1)==size(gt0T,1));
  for i=1:n, [gtM{i},dtM{i}] = evalResMulti(gt0C{i},gt0T{i},dt0C{i},dt0T{i},thr,mul); end; return;
end

% check inputs
if(isempty(gt0C)), gt0C=zeros(0,5); end
if(isempty(gt0T)), gt0T=zeros(0,5); end
if(isempty(dt0C)), dt0C=zeros(0,5); end
if(isempty(dt0T)), dt0T=zeros(0,5); end
assert( size(dt0C,2)==5 );
assert( size(dt0T,2)==5 );
nd=size(dt0T,1);
assert( size(gt0C,2)==5 );
assert( size(gt0T,2)==5 );
ng=size(gt0T,1);

assert(size(gt0C,1)==size(gt0T,1));

% if either is ignored, ignore both
gtCT = gt0C(:,5) | gt0T(:,5);
gt0C(:,5)=gtCT;
gt0T(:,5)=gtCT;
% sort dt highest score first, sort gt ignore last
[~,ord]=sort(dt0T(:,5),'descend');
dt0C=dt0C(ord,:);
dt0T=dt0T(ord,:);
[~,ord]=sort(gtCT,'ascend');
gt0C=gt0C(ord,:);
gt0T=gt0T(ord,:);

gtC=gt0C; gtC(:,5)=-gtC(:,5);
gtT=gt0T; gtT(:,5)=-gtT(:,5);
dtC=dt0C; dtC=[dtC zeros(nd,1)];
dtT=dt0T; dtT=[dtT zeros(nd,1)];

% Attempt to match each (sorted) dt to each (sorted) gt
oa = compOasMulti( dtC(:,1:4), dtT(:,1:4), gtC(:,1:4), gtT(:,1:4), gtT(:,5)==-1 );

gtM=gtT; dtM=dtT; 
for d=1:nd
  bstOa=thr; bstg=0; bstm=0; % info about best match so far
  for g=1:ng
    % if this gt already matched, continue to next gt
    m=gtM(g,5); if( m==1 && ~mul ), continue; end
    % if dt already matched, and on ignore gt, nothing more to do
    if( bstm~=0 && m==-1 ), break; end
    % compute overlap area, continue to next gt unless better match made
    if(oa(d,g)<bstOa), continue; end
    % match successful and best so far, store appropriately
    bstOa=oa(d,g); bstg=g; if(m==0), bstm=1; else bstm=-1; end
  end; g=bstg; m=bstm;
  % store type of match for both dt and gt
  if(m==-1), dtM(d,6)=m; elseif(m==1), gtM(g,5)=m; dtM(d,6)=m; end
end

end


function oa = compOasMulti( dtC, dtT, gtC, gtT, ig )
% Computes (modified) overlap area between pairs of bbs.
%
% Uses modified Pascal criteria with "ignore" regions. The overlap area
% (oa) of a ground truth (gt) and detected (dt) bb is defined as:
%  oa(gt,dt) = area(intersect(dt,dt)) / area(union(gt,dt))
% In the modified criteria, a gt bb may be marked as "ignore", in which
% case the dt bb can can match any subregion of the gt bb. Choosing gt' in
% gt that most closely matches dt can be done using gt'=intersect(dt,gt).
% Computing oa(gt',dt) is equivalent to:
%  oa'(gt,dt) = area(intersect(gt,dt)) / area(dt)
%
% USAGE
%  oa = bbGt( 'compOas', dt, gt, [ig] )
%
% INPUTS
%  dt       - [mx4] detected bbs
%  gt       - [nx4] gt bbs
%  ig       - [nx1] 0/1 ignore flags (0 by default)
%
% OUTPUTS
%  oas      - [m x n] overlap area between each gt and each dt bb
%
% EXAMPLE
%  dt=[0 0 10 10]; gt=[0 0 20 20];
%  oa0 = bbGt('compOas',dt,gt,0)
%  oa1 = bbGt('compOas',dt,gt,1)
%
% See also bbGt, bbGt>evalRes
if ne(size(dtC,1),size(dtT,1))
    print('Error, unequal detection bounding boxes between modalities.');
    return; 
end
if ne(size(gtC,1),size(gtT,1))
    print('Error, unequal groundtruth bounding boxes between modalities.');
    return; 
end
m=size(dtT,1); n=size(gtT,1); oa=zeros(m,n);
if(nargin<5), ig=zeros(n,1); end
deC=dtC(:,[1 2])+dtC(:,[3 4]); daC=dtC(:,3).*dtC(:,4);
deT=dtT(:,[1 2])+dtT(:,[3 4]); daT=dtT(:,3).*dtT(:,4);
geC=gtC(:,[1 2])+gtC(:,[3 4]); gaC=gtC(:,3).*gtC(:,4);
geT=gtT(:,[1 2])+gtT(:,[3 4]); gaT=gtT(:,3).*gtT(:,4);
for i=1:m
  for j=1:n
    wC=min(deC(i,1),geC(j,1))-max(dtC(i,1),gtC(j,1)); if(wC<=0), continue; end
    wT=min(deT(i,1),geT(j,1))-max(dtT(i,1),gtT(j,1)); if(wT<=0), continue; end
    hC=min(deC(i,2),geC(j,2))-max(dtC(i,2),gtC(j,2)); if(hC<=0), continue; end
    hT=min(deT(i,2),geT(j,2))-max(dtT(i,2),gtT(j,2)); if(hT<=0), continue; end
    tC=wC*hC;
    tT=wT*hT;
    if(ig(j))
        u=daC(i)+daT(i); 
    else
        u=daC(i)+gaC(j)-tC+daT(i)+gaT(j)-tT;
    end
    oa(i,j)=(tC+tT)/u;
  end
end
end