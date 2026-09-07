from pathlib import Path
from datetime import datetime
import time

from modules.context import context
from modules.config import Config
from modules.config.schemas_v1 import Logging, LoggingSavePK3
from modules.game import set_rom
from modules.libmgba import LibmgbaEmulator
from modules.modes import get_bot_listeners
from modules.profiles import Profile
from modules.save_import import guess_rom_from_save_state
from modules.modes.campaign import CampaignProgressionMode
from modules.modes.opening import EmeraldOpeningCapability, OpeningSequenceState, _scripted_std_msgbox_waiting
from modules.memory import get_game_state, GameState
from modules.player import get_player_avatar, player_avatar_is_controllable
from modules.tasks import (
    get_tasks,
    get_global_script_context,
    is_waiting_for_input,
    is_field_message_waiting_for_input,
    get_text_printer,
)
from modules.map import get_event_flag

from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.persistence import JsonEventStore

def run_diagnostic():
    state_file = Path("profiles/test_truck_scene/current_state.ss1")
    with open(state_file, "rb") as handle:
        rom, state_data, save_data = guess_rom_from_save_state(handle, None)
    
    profile = Profile(rom, Path("profiles/test_truck_scene"), datetime.now())
    context.testing = True
    context.config = Config(Path("modules/config/templates"))
    context.config.logging = Logging(
        save_pk3=LoggingSavePK3(shiny=False, custom=False, roamer=False),
        log_encounters=True,
        log_encounters_to_console=False,
        desktop_notifications=False,
        shiny_gifs=False,
        tcg_cards=False,
    )
    context.profile = profile
    set_rom(profile.rom)
    
    def do_nothing():
        pass
        
    context.emulator = LibmgbaEmulator(context.profile, do_nothing, is_test_run=True)
    context.emulator.set_audio_enabled(False)
    context.emulator.set_video_enabled(False)
    context.emulator.set_throttle(False)
    if save_data is not None:
        context.emulator.load_save_game(save_data)
    context.emulator.load_save_state(state_data)
    
    from modules.memory import write_to_save_block
    write_to_save_block(bytes([2]), num=2, offset=0x14)
    
    context.bot_listeners = get_bot_listeners(context.rom)
    context.nuzlocke_runtime = NuzlockeRuntime(event_sink=JsonEventStore(context.profile.path / "nuzlocke_events.json"))
    context.frame = 0
    context.controller_stack.clear()
    
    mode = CampaignProgressionMode()
    gen = mode.run()
    context.controller_stack.append(gen)
    
    print("Starting diagnostic run from test_truck_scene...")
    
    frame_count = 0
    start_time = time.perf_counter()
    
    last_print = 0
    clock_set_frame = None
    
    last_map = None
    last_phase = None
    for _ in range(12000):
        frame_count += 1
        context.frame += 1
        context.nuzlocke_runtime.update()
        
        avatar = get_player_avatar()
        coords = avatar.local_coordinates if avatar else None
        map_id = avatar.map_group_and_number if avatar else None
        controllable = player_avatar_is_controllable()
        game_state = get_game_state()
        script_ctx = get_global_script_context()
        active_tasks = [t.symbol for t in get_tasks()] if get_tasks() else []
        clock_flag = get_event_flag("SET_WALL_CLOCK")
        
        inputs = context.emulator.get_inputs()
        text_printer = get_text_printer()
        waiting = is_waiting_for_input()
        field_waiting = is_field_message_waiting_for_input()

        try:
            next(gen)
        except StopIteration:
            print(f"Campaign generator finished at frame {frame_count}")
            break
        except Exception as e:
            print(f"Exception at frame {frame_count}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break
            
        inputs_after = context.emulator.get_inputs()
        
        if frame_count % 200 == 0 or map_id != last_map:
            print(f"F{frame_count}: map={map_id} pos={coords} ctrl={controllable} waiting={waiting} native={getattr(script_ctx, 'native_function_name', None)} script={getattr(script_ctx, 'script_function_name', None)} tasks={active_tasks} inputs={inputs}")
            last_map = map_id
            
        context.emulator.run_single_frame()
        
        if clock_set_frame is not None and (frame_count - clock_set_frame) > 100:
            print("Captured 100 frames post-clock, stopping.")
            break

    elapsed = time.perf_counter() - start_time
    print(f"Run completed: {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.1f} fps)")

if __name__ == "__main__":
    run_diagnostic()
