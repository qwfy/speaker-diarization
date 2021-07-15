this exp basically corresponding to the best case scenario:

- ref rttm is a merge of xf transcript and human label
- all sentences shorter than 1.5s is dropped from rttm
- ref vad is translated directly from the ref rttm
- overlapping voice intervals are merged
  (even they are of different speaker)
- VBx vanilla clustering is used
  (i.e. no fancy merging at the end, only the first label is used)

ref rttm: result/label_rttm_xf
ref vad: result/vad_xf
