import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.curation as sc
import spikeinterface.widgets as sw
import numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import npc_session
import tqdm

from export_to_phy import *


def main():

    params = Params()
    session_id = get_single_session_id()
    print(session_id)

    # raw data
    ecephys_folder = data_folder / session_id
    ecephys_compressed_folder = get_ecephys_folder(session_id, 'compressed')
    ecephys_clipped_folder = get_ecephys_folder(session_id, 'clipped')
    raw_compressed_folder = ecephys_compressed_folder

    # sorted data
    sorted_folder = get_sorted_folder(session_id)
    postprocessed_folder = sorted_folder / "postprocessed"
    curated_folder = sorted_folder / "curated"
    pre_curated_folder = sorted_folder / "sorting_precurated"
    spikesorted_folder = sorted_folder / "spikesorted"

    is_sorting_analyzer = bool(next(spikesorted_folder.glob("*/*.zarr"), None))


    # loop through experiments and segments to get the longest one
    lengths = []
    experiment_inds = {re.search(r'experiment(\d+)', p.name)[1] for p in raw_compressed_folder.iterdir()}
    segment_inds = []

    for exp_ind in experiment_inds:
        # any probe will do:
        rec_path = sorted(p for p in raw_compressed_folder.iterdir() if p.name.endswith('-AP.zarr') and f'experiment{exp_ind}' in p.name)[0]
        rec = zarr.open(rec_path)
        for k, v in rec.items():
            if not k.startswith('traces_seg'):
                continue
            traces_seg = int(k.split('traces_seg')[-1])
            lengths.append(
                {
                    'experiment': exp_ind,
                    'traces_seg': traces_seg,
                    'recording': traces_seg + 1, # spikeinterface refers to recordings as 1-indexed
                    'len': len(v),
                }
            )

    longest = sorted(lengths, key=lambda x: x['len'])[-1]

    print(f'Selected experiment{longest["experiment"]} recording{longest["recording"]} (approx {longest["len"]/30_000/3600 :.1f} hr)')

    # stream name: 
    streams =  sorted(
        [
            p
            for p in postprocessed_folder.iterdir() 
            if p.is_dir() and "post" not in p.name
            and f'experiment{longest["experiment"]}' in p.name   
            and f'recording{longest["recording"]}' in p.name 
        ],
        key=lambda p: p.name.split('#')[-1],
    )
    colors=["C0", "C1"]
    results_folder = Path('/root/capsule/results/plots')
    results_folder.mkdir(exist_ok=True, parents=True)

    for stream in streams:
        stream_name = stream.name.removesuffix('.zarr')
        probe = npc_session.ProbeRecord(stream.name)

        if is_sorting_analyzer:
            we = si.load(postprocessed_folder / stream.name)
        else:
            we = si.load_waveforms(postprocessed_folder / stream.name, with_recording=False)

        # potential_merges = [(173, 413)]
        # load recording
        recording = si.read_zarr(next(p for p in ecephys_compressed_folder.iterdir() if stream_name.split('_recording')[0] in p.name))

        recording = recording.select_segments(longest['traces_seg'])
        
        if is_sorting_analyzer:
            good_channel_ids = recording.channel_ids[np.isin(recording.channel_ids, we.channel_ids)]
            recording = recording.select_channels(good_channel_ids)
        else:
            channels = recording.channel_ids[np.isin(recording.channel_ids, we.channel_ids)]
            if hasattr(recording, 'channel_slice'):
                recording = recording.channel_slice(channels)
            elif hasattr(recording, 'select_channels'):
                recording = recording.select_channels(channels)

        recording = spre.common_reference(recording)
        recording = spre.phase_shift(recording)
        if params.filter_type == 'highpass': 
            recording = spre.highpass_filter(recording)      
        elif params.filter_type == 'bandpass':
            recording = spre.bandpass_filter(recording)
        else:
            raise ValueError(f"'filter_type' must be one of ['highpass', 'bandpass'], not {params.filter_type!r}")

        if is_sorting_analyzer:
            we.set_temporary_recording(recording.select_segments(longest['traces_seg']-1))
        else:
            we.set_recording(recording)

        steps = ['remove_contaminated', 'template_similarity', 'maximum_distance_um', 'minimum_spikes']
        #['min_spikes', 'remove_contaminated', 'unit_positions', 'correlogram', 'template_similarity']
        potential_merges = sc.get_potential_auto_merge(we, steps=steps, maximum_distance_um=40, minimum_spikes=200)
        
        for units in tqdm.tqdm(potential_merges, desc='plotting'):
            label = f'probe{probe}_{units[0]}-{units[1]}'
            title = we.folder.as_posix().removeprefix('/root/capsule/data/ecephys_')
            colors_dict = dict(zip(units, colors))
            
            sw.plot_amplitudes(
                we, 
                unit_ids=units, 
                unit_colors=colors_dict, 
                max_spikes_per_unit=None,
                plot_legend=False,
            )
            plt.gcf().suptitle(title, fontsize=6)
            plt.gca().legend(loc='lower center', ncol=2)
            plt.savefig(results_folder / f"{label}_amplitudes.png")
            plt.tight_layout()
            plt.close()

            sw.plot_crosscorrelograms(
                we, 
                unit_ids=units, 
                unit_colors=colors_dict,
            )
            plt.gcf().suptitle(title, fontsize=6)
            plt.tight_layout()
            plt.savefig(results_folder / f"{label}_ccg.png")
            plt.close()

            w = sw.plot_unit_templates(we, unit_ids=units, unit_colors=colors_dict, plot_legend=True, lw_templates=1)
            plt.gcf().suptitle(title, fontsize=6)
            plt.tight_layout()
            plt.savefig(results_folder / f"{label}_templates.png")
            plt.close()

if __name__ == "__main__":
    main()