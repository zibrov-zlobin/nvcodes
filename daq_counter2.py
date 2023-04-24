# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:18:51 2023

@author: notsowells
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import nidaqmx

from typing import Any, Sequence, Union, Dict, Optional
from qcodes.instrument import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.parameter import Parameter, ParameterWithSetpoints
from qcodes.validators import Arrays, Numbers

from random import randint


class CountAxis(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        return np.array(self.root_instrument.npts.get_latest())

class DAQCounts(ParameterWithSetpoints): # this is the more kosher way of doing things apparently

    def get_raw(self):
        task = self.root_instrument.task
        data_raw = np.diff(np.array(task.read(number_of_samples_per_channel=self.root_instrument.samples_to_read+1,
                                      timeout=60)))
        return data_raw
    
class DAQCounter(Instrument): # replace with instrument channel later
    """ InstrumentChannel to acquire DAQ counter inputs in a qcodes measurement

    Args:
        name: Name of instrumentchannel (usually, 'daq_ctr')
        dev_name: NI DAQ device name (e.g. 'Dev1')
        channel: channel name (e.g. 'ctr0')
        config: configuration dictionary
        timing: ???? Should there a be defualt to sample clock?
    """
    def __init__(self,
                    name: str,
                    dev_name: str,
                    channel: str,
                    clksource: str,
                    sampling_rate: int,
                    sample_mode: Optional[str]='continious',
                    samples_to_read: Optional[int]=1000,
                    timeout: Optional[int] =60,
                    config: Optional[dict[str, Any]]=None,
                    **kwargs) -> None:
        
        super().__init__(name, **kwargs)

        if config is not None:
            config = self.parse_config(config)
        else:
            config = {'counter': dev_name+"/"+channel,
                'name_to_assign_to_channel': 'counter'+str(randint(1,100)),
                'edge': nidaqmx.constants.Edge.RISING,
                'initial_count': 0,
                'count_direction': nidaqmx.constants.CountDirection.COUNT_UP
                }
            
        self.running = False
        self.samples_to_read = samples_to_read
        self.timeout = timeout
        self.clksource = clksource

        self.add_parameter(
            "run",
            label="Clock State",
            get_cmd = self.state,
            set_cmd = lambda x: self.start() if x else self.stop(),
            val_mapping={
                "off":0,
                "on": 1,
                }
            )
        
        self.metadata.update(config)
        self.task = nidaqmx.Task("ctr"+str(randint(1,100)))
        self.task.ci_channels.add_ci_count_edges_chan(**config)
        if sample_mode == 'continious':
            print(self._get_ctr_string(clksource))
            self.task.timing.cfg_samp_clk_timing(source= self._get_ctr_string(clksource),
                                                rate= sampling_rate,
                                                active_edge= nidaqmx.constants.Edge.RISING,
                                                sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                samps_per_chan= int(samples_to_read)
                                                )
        else:
            pass
        
        self.add_parameter(
            name='npts',
            initial_value=self.samples_to_read,
            label='Number of points',
            get_cmd=None,
            set_cmd=None,
            docstring="mindless copying"
        )
        self.add_parameter(
            name='timing',
            parameter_class=CountAxis,
            vals=Arrays(shape=(self.npts.get_latest, )),
        )
        self.add_parameter(
            name='counts',
            parameter_class=DAQCounts,
            setpoints= (self.timing,),
            vals=Arrays(shape=(self.npts.get_latest, ))
        )

    @staticmethod
    def _get_ctr_string(s:str) -> str:
        t = s.split('/')
        return t[-1].capitalize()+"InternalOutput"

    def get_idn(self) -> None:
        pass

    def parse_config(self) -> None:
        pass

    def start(self) -> None:
        self.task.start()
        self.running = True
    
    def stop(self) -> None:
        self.task.stop()
        self.running = False
    
    def state(self) -> None:
        return self.running

    def clear_task(self) -> None:
        self.task.close()
    
    def close(self) -> None:
        self.clear_task()

class DAQCOChannel(Instrument): #for some reason doesnt work with InstrumentChannel. __super__(name) is where it hiccups
    def __init__(self,
                 name: str,
                 dev_name: str,
                 channel: str,
                 sample_rate: int,
                 samples_per_chan: int,
                 config: Optional[dict]=None,
                 **kwargs) -> None:
        super().__init__(name, **kwargs)

        if config is not None:
            config = self.parse_config(config)
        else:
            config = {'counter': dev_name+"/"+channel,
                'name_to_assign_to_channel': 'clock'+str(randint(1,100)),
                'units': nidaqmx.constants.FrequencyUnits.HZ,
                'idle_state': nidaqmx.constants.Level.LOW,
                'initial_delay': 0.0,
                'freq': sample_rate,
                'duty_cycle': 0.5}
            
        self.running = False
        self.add_parameter(
            "run",
            label="Clock State",
            get_cmd = self.state,
            set_cmd = lambda x: self.start() if x else self.stop(),
            val_mapping={
                "off":0,
                "on": 1,
                }
            )
        
        self.metadata.update(config)
        self.task = nidaqmx.Task("clk"+str(randint(1,100)))
        self.task.co_channels.add_co_pulse_chan_freq(**config)
        self.task.timing.cfg_implicit_timing(sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                             samps_per_chan= int(samples_per_chan))
    
    def parse_config(self, config):
        """ Validates config settings

        Args:
            config (dict): contains clock configuration settings
        """
        pass

    def start(self) -> None:
        self.task.start()
        self.running = True
    
    def stop(self) -> None:
        self.task.stop()
        self.running = False
    
    def state(self) -> None:
        return self.running

    def clear_task(self) -> None:
        self.task.close()
    
    def close(self) -> None:
        self.clear_task()

class DAQAnalogOutputVoltage(Parameter):
    """Writes data to one or several DAQ analog outputs. This only writes one channel at a time,
    since Qcodes ArrayParameters are not settable.
    Args:
        name: Name of parameter (usually 'voltage').
        dev_name: DAQ device name (e.g. 'Dev1').
        idx: AO channel index.
        kwargs: Keyword arguments to be passed to ArrayParameter constructor.
    """
    def __init__(self, name:str, chan_name: str) -> None:
        super().__init__(name=name, label='Voltage', unit='V')
        self._voltage = np.nan
        self.channel = chan_name
    
    def set_raw(self, voltage: Union[int, float]) -> None:
        with nidaqmx.Task('daq_ao_task') as ao_task:
            ao_task.ao_channels.add_ao_voltage_chan(self.channel) # maybe add a unique name (channel, unique_name)
            ao_task.write(voltage, auto_start=True)
        self._voltage = voltage
    
    def get_raw(self): # I think this can be readout. 
        """Returns last voltage array written to outputs
        """
        return self._voltage

class DAQAOChannel(InstrumentChannel):
    """Instrument to write DAQ analog output data in a qcodes Loop or measurement.
    Args:
        name: Name of instrument (usually 'daq_ao').
        dev_name: NI DAQ device name (e.g. 'Dev1').
        channels: Dict of analog output channel configuration.
        **kwargs: Keyword arguments to be passed to Instrument constructor.
    """
    def __init__(self, parent: 'NIDAQ', name: str, channum: int):
        super().__init__(parent, name)
        self._parent = parent
        self.chan_name = f'{self._parent.dev_name}/ao{channum}'
        self.voltage = DAQAnalogOutputVoltage('voltage', self.chan_name)

class DAQAIChannel(InstrumentChannel):
    pass


class Sweep_Context:
    """ need to setup sweep voltages, clock, counter, and analog output write
    """
    def __init__(self, sweep: np.ndarray):
        self.sweep = sweep

    def __enter__(self):
        return self
    
    def __exit__(self):
        # clocked context should take care of stopping the clock
        return False
    
    def start(self) -> None:
        


class NIDAQClocked_Context:
    def __init__(self, daq: 'NIDAQ', clksettings: Dict[str, Any]):
        self._parent = daq
        self._parent._set_up_clock(clksettings)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._parent.clear_clock(self)
        return False
    
    def sweep1d() -> Sweep_Context:
        pass

    def sweep2d() -> Sweep_Context:
        pass
    

class NIDAQ(Instrument):
    def __init__(self, name:str, dev_name:str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.dev_name = dev_name
        self._task_list = []
        self._clktask = None
        self._set_up_aochannels()

    def _set_up_clock(self, clk_settings):
        self._clktask = nidaqmx.Task('clock')
        self._clktask.co_channels.add_co_pulse_chan_freq()
        self._clktask.timing.cfg_implicit_timing()

    def _clear_clock(self) -> None:
        self._clktask.stop()
        self._clktack.close()
    
    # def sweep(self, output:str, voltages: Sequence[float],)

    def _set_up_aochannels(self) -> None:
        channels = ChannelList(self, 'AOChannels', DAQAOChannel, snapshotable=False)
        for i in range(0, 8):
            name = f'ao{i}'
            channel = DAQAOChannel(self, name, i)
            self.add_submodule(name, channel)
            channels.append(channel)
        channels.lock()
        self.add_submodule('aochannels', channels)
