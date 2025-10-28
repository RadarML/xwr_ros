std_msgs/Header header
std_msgs/Int16MultiArray iq # raw data contain chirp layout

float32 frequency       # GHz
float32 idle_time       # us
float32 adc_start_time  # us
float32 ramp_end_time   # us
float32 tx_start_time   # us
float32 freq_slope      # MHz/us
uint16 adc_samples      #
uint16 sample_rate      #
uint16 frame_length     # ms