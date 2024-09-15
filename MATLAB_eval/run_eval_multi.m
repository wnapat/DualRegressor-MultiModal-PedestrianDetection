% testing ground truth and detection results
gtDir = 'data\kaist-rgbt\annotations';
testSet = 'data\kaist-rgbt\splits\test-all-20.txt';

dets_filename_vis = 'data\kaist-rgbt\dets\ours\det_vis';
dets_filename_lwir = 'data\kaist-rgbt\dets\ours\det_lwir';

% bounding box overlap threshold and shift 
threshold = 0.5;
shift = 0;

% parameters
pLoad=[{'lbls',{'person'}, 'ilbls',{'people','person?','cyclist'}},...
  'hRng',[55 inf], 'vType',{{'none','partial'}},...
  'xRng',[5 635], 'yRng',[5 507], 'format', 0];

% evaluate detector and plot roc miss rate
[miss,roc,~]=kaistEvalMulti('nameC',dets_filename_vis, 'nameT',dets_filename_lwir,...
  'gtDir',gtDir, 'threshold', threshold, 'shift',shift, 'pLoad',pLoad, ...
  'subset',testSet, 'inputType',{'visible', 'lwir'}, ...
  'show',1, 'clr','red');

fprintf('threshold: %02.2f, shift distance: %d, Multi-modal MR = %02.2f\n', threshold, shift, miss*100);