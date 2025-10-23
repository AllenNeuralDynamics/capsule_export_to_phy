from pathlib import Path
import os
import pprint  
import re
import shutil
import warnings
from typing import Literal

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.exporters as sexp
import numpy as np
import pandas as pd
import aind_session 
import zarr
import pydantic_settings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class Params(pydantic_settings.BaseSettings):
    
    filter_type: Literal['highpass', 'bandpass'] = 'highpass'
    compute_pc_features: bool = True
    n_cpus: int = int(os.getenv('CO_CPUS', 0)) - 1

    # set the priority for auto-parsing sources:
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # the order of the sources is what defines the priority:
        # - first source is highest priority
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='parameters.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )

data_folder = Path("/root/capsule/data")
scratch_folder = Path("/root/capsule/scratch")
results_folder = Path("/root/capsule/results")

def get_sessions() -> set[aind_session.Session]:
    sessions = []
    for p in data_folder.iterdir():
        try:
            sessions.append(aind_session.Session(p.name))
        except ValueError:
            continue
    return set(sessions)

def get_raw_folder(session_id: str) -> Path:
    try:
        return next(
            p for p in data_folder.iterdir() 
            if p.is_dir() and "sorted" not in p.name and session_id in p.name
        )
    except StopIteration:
        raise FileNotFoundError(f"No raw data folder found for {session_id}") from None

def get_ecephys_folder(session_id: str, folder_name: Literal['compressed', 'clipped']) -> Path:
    options = ('compressed', 'clipped')
    if folder_name not in options:
        raise ValueError(f"'folder_name' must be one of {options!r}, not {folder_name!r}")
    raw = get_raw_folder(session_id)
    # may be in an ecephys modality subfolder
    for p in (f'ecephys_{folder_name}', f'ecephys/ecephys_{folder_name}'):
        if (candidate := raw / p).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate clipped ecephys folder in {raw}")

def get_sorted_folder(session_id: str) -> Path:
    try:
        return next(
            p for p in data_folder.iterdir() 
            if p.is_dir() and "sorted" in p.name and "spikesorted" not in p.name and session_id in p.name
        )
    except StopIteration:
        raise FileNotFoundError(f"No sorted data folder found for {session_id}") from None

def get_single_session_id(use_oldest: bool = False) -> str:
    ids = sorted([s.id for s in get_sessions()])
    if (n := len(ids)) > 1 and not use_oldest:
        raise OSError(f'Found data for {n} sessions. Attach raw + sorted assets for only 1 session')
    elif n > 1:
        print(f"Data for multiple sessions found: using oldest")
    elif n == 0:
        raise FileNotFoundError(f'No data found. Attach raw + sorted assets for 1 session')
    return ids[0]


def process_session(session_id: str, params: Params) -> None:
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

    for stream in streams:
        stream_name = stream.name.removesuffix('.zarr')

        # save location
        phy_folder = results_folder / f"{sorted_folder.name}_phy" / stream_name
        phy_folder.mkdir(exist_ok=True, parents=True)
        print(f'Processing {phy_folder}')

        if is_sorting_analyzer:
            we_recless = si.load(postprocessed_folder / stream.name)
        else:
            we_recless = si.load_waveforms(postprocessed_folder / stream.name, with_recording=False)

        # load recording
        recording = si.read_zarr(next(p for p in ecephys_compressed_folder.iterdir() if stream_name.split('_recording')[0] in p.name))

        recording = recording.select_segments(longest['traces_seg'])
        
        if is_sorting_analyzer:
            good_channel_ids = recording.channel_ids[np.isin(recording.channel_ids, we_recless.channel_ids)]
            recording = recording.select_channels(good_channel_ids)
        else:
            good_channel_mask = np.isin(recording.channel_ids, we_recless.channel_ids)
            recording = recording.channel_slice(recording.channel_ids[good_channel_mask])

        recording = spre.common_reference(recording)
        recording = spre.phase_shift(recording)
        if params.filter_type == 'highpass': 
            recording = spre.highpass_filter(recording)      
        elif params.filter_type == 'bandpass':
            recording = spre.bandpass_filter(recording)
        else:
            raise ValueError(f"'filter_type' must be one of ['highpass', 'bandpass'], not {params.filter_type!r}")

        if is_sorting_analyzer:
            we_recless.set_temporary_recording(recording.select_segments(longest['traces_seg']-1))
        else:
            we_recless.set_recording(recording)

        # save
        sexp.export_to_phy(
            we_recless, 
            output_folder=phy_folder,
            compute_pc_features=params.compute_pc_features, 
            remove_if_exists=True,
            copy_binary=False,
            n_jobs=params.n_cpus,
            progress_bar=True,
            chunk_duration="1s",
            dtype='int16',
        )

        spike_locations = we_recless.load_extension("spike_locations").get_data()
        spike_depths = spike_locations["y"]
        np.save(phy_folder / "spike.depths.npy", spike_depths)

        # save decoder
        if not os.path.exists(curated_folder):
            curation_dir = f'{pre_curated_folder}/{stream_name}/properties'
        else:
            curation_dir = f'{curated_folder}/{stream_name}/properties'
        unit_ids = we_recless.unit_ids
        for file in os.listdir(curation_dir):
            if '.npy' in file:
                qc_name = file.split('.npy')[0]
                qc_value = np.load(f"{curation_dir}/{file}", allow_pickle=True)
                # save metrics
                metric = pd.DataFrame(
                    {"cluster_id": [i for i in range(len(unit_ids))], qc_name: qc_value}
                )     
                metric.to_csv(phy_folder / f"cluster_{qc_name}.tsv", sep="\t", index=False)           

        print(f'Finished {phy_folder}.')
    print(f'Finished {session_id}')

if __name__ == "__main__":
    params = Params()
    print('Using params:')
    pprint.pprint(params.model_dump())
    
    session_id = get_single_session_id(use_oldest=True)
    process_session(session_id, params)