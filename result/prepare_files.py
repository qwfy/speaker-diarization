if __name__ == '__main__':
  import os
  import glob
  import subprocess
  import result.rttm

  raw_audios = list(glob.glob('result/audio_raw/*.*'))

  for raw in raw_audios:
    print('processing', raw)
    basename = os.path.basename(raw)
    basename, _ext = os.path.splitext(basename)

    # converting audio to wav
    wav_path = f'result/audio_wav/{basename}.wav'
    print('==> converting to', wav_path)
    # subprocess.run(
    #   ['ffmpeg', '-loglevel', 'warning', '-hide_banner', '-y',
    #    '-i', raw, '-ar', '16000', wav_path],
    #   check=True
    # )
    subprocess.run(
      ['sox', raw, '-r', '16000', wav_path, 'remix', '1,2']
    )

    # converting audio label to rttm
    tsv_path = f'result/label_txt/{basename}.txt'
    raw_rttm_path = f'result/label_rttm_raw/{basename}.rttm'
    print('==> converting to', raw_rttm_path)
    result.rttm.convert_tsv_to_rttm(tsv_path, raw_rttm_path)

    # merge rttm by adjacency
    adjacency_merged = f'result/label_rttm_adjacent_merged/{basename}.rttm'
    print('==> converting to', adjacency_merged)
    result.rttm.merge_rttm_by_adjacency(raw_rttm_path, adjacency_merged)
    result.rttm.rttm_to_tsv(adjacency_merged, f'{adjacency_merged}.txt')

    st_merged = f'result/label_rttm_st_merged/{basename}.rttm'
    print('==> converting to', st_merged)
    result.rttm.merge_rttm_by_st(adjacency_merged, st_merged)
    result.rttm.rttm_to_tsv(st_merged, f'{st_merged}.txt')

    # write oracle vad
    # here we use the adjacency merged version,
    # to avoid variate length of silence in the labeling,
    # ideally we should use the unmerged one
    vad_path = f'result/vad_oracle/{basename}.lab'
    print('==> writing oracle vad', vad_path)
    result.rttm.rttm_to_oracle_vad(adjacency_merged, vad_path)

    print('==> done')
