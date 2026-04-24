std_msgs/Header header
std_msgs/String device
float32 frequency       # GHz
float32 idle_time       # us
float32 adc_start_time  # us
float32 ramp_end_time   # us
float32 tx_start_time   # us
float32 freq_slope      # MHz/us
uint16 adc_samples      # number of samples per chirp
uint16 sample_rate      # KHz
uint16 frame_length     # number of chirp loop
float32 frame_period    # ms