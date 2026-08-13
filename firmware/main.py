from machine import Pin, ADC, PWM, I2C
import dht
import time

i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=100000)

BMP180_ADDR = 0x77


def _s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v > 32767 else v


class BMP180:
    def __init__(self, i2c):
        self.i2c = i2c
        self.ok = False
        try:
            d = i2c.readfrom_mem(BMP180_ADDR, 0xAA, 22)
            self.AC1 = _s16(d[0], d[1]);  self.AC2 = _s16(d[2], d[3])
            self.AC3 = _s16(d[4], d[5])
            self.AC4 = (d[6] << 8) | d[7]
            self.AC5 = (d[8] << 8) | d[9]
            self.AC6 = (d[10] << 8) | d[11]
            self.B1 = _s16(d[12], d[13]); self.B2 = _s16(d[14], d[15])
            self.MB = _s16(d[16], d[17]); self.MC = _s16(d[18], d[19])
            self.MD = _s16(d[20], d[21])
            self.ok = True
        except Exception:
            self.ok = False

    def _raw_temp(self):
        self.i2c.writeto_mem(BMP180_ADDR, 0xF4, bytes([0x2E]))
        time.sleep_ms(5)
        d = self.i2c.readfrom_mem(BMP180_ADDR, 0xF6, 2)
        return (d[0] << 8) | d[1]

    def _raw_pressure(self):
        self.i2c.writeto_mem(BMP180_ADDR, 0xF4, bytes([0x34]))
        time.sleep_ms(8)
        d = self.i2c.readfrom_mem(BMP180_ADDR, 0xF6, 3)
        return ((d[0] << 16) + (d[1] << 8) + d[2]) >> 8

    def read_hpa(self):
        if not self.ok:
            return 1013.25
        try:
            ut = self._raw_temp()
            up = self._raw_pressure()
            x1 = ((ut - self.AC6) * self.AC5) >> 15
            x2 = (self.MC << 11) // (x1 + self.MD)
            b5 = x1 + x2
            b6 = b5 - 4000
            x1 = (self.B2 * (b6 * b6 >> 12)) >> 11
            x2 = (self.AC2 * b6) >> 11
            x3 = x1 + x2
            b3 = (((self.AC1 * 4 + x3) << 0) + 2) >> 2
            x1 = (self.AC3 * b6) >> 13
            x2 = (self.B1 * (b6 * b6 >> 12)) >> 16
            x3 = ((x1 + x2) + 2) >> 2
            b4 = (self.AC4 * (x3 + 32768)) >> 15
            b7 = (up - b3) * 50000
            p = (b7 * 2) // b4 if b7 < 0x80000000 else (b7 // b4) * 2
            x1 = (p >> 8) * (p >> 8)
            x1 = (x1 * 3038) >> 16
            x2 = (-7357 * p) >> 16
            p = p + ((x1 + x2 + 3791) >> 4)
            return p / 100.0
        except Exception:
            return 1013.25


LCD_ADDR = 0x27


class I2cLcd:
    def __init__(self, i2c, addr=LCD_ADDR):
        self.i2c = i2c
        self.addr = addr
        self.backlight = 0x08
        self.ok = False
        try:
            time.sleep_ms(50)
            for _ in range(3):
                self._write4(0x03, 0)
                time.sleep_ms(5)
            self._write4(0x02, 0)
            self._cmd(0x28)
            self._cmd(0x0C)
            self._cmd(0x06)
            self.clear()
            self.ok = True
        except Exception:
            self.ok = False

    def _strobe(self, data):
        self.i2c.writeto(self.addr, bytes([data | 0x04 | self.backlight]))
        time.sleep_us(1)
        self.i2c.writeto(self.addr, bytes([(data & 0xFB) | self.backlight]))
        time.sleep_us(100)

    def _write4(self, nibble, rs):
        data = ((nibble << 4) & 0xF0) | rs
        self.i2c.writeto(self.addr, bytes([data | self.backlight]))
        self._strobe(data)

    def _cmd(self, cmd):
        self._write4(cmd >> 4, 0)
        self._write4(cmd & 0x0F, 0)

    def _data(self, val):
        self._write4(val >> 4, 1)
        self._write4(val & 0x0F, 1)

    def clear(self):
        self._cmd(0x01)
        time.sleep_ms(2)

    def move_to(self, col, row):
        offsets = (0x00, 0x40)
        self._cmd(0x80 | (offsets[row] + col))

    def show(self, line0, line1):
        if not self.ok:
            return
        try:
            self.move_to(0, 0)
            for ch in line0[:16].ljust(16):
                self._data(ord(ch))
            self.move_to(0, 1)
            for ch in line1[:16].ljust(16):
                self._data(ord(ch))
        except Exception:
            pass


bmp = BMP180(i2c)
lcd = I2cLcd(i2c)

dht_sensor = dht.DHT22(Pin(15))

trig = Pin(5, Pin.OUT)
echo = Pin(18, Pin.IN)

pot_rain = ADC(Pin(34)); pot_rain.atten(ADC.ATTN_11DB)
pot_soil = ADC(Pin(35)); pot_soil.atten(ADC.ATTN_11DB)

float_low = Pin(32, Pin.IN, Pin.PULL_DOWN)
float_mid = Pin(33, Pin.IN, Pin.PULL_DOWN)
float_high = Pin(25, Pin.IN, Pin.PULL_DOWN)

led_green = Pin(26, Pin.OUT)
led_yellow = Pin(27, Pin.OUT)
led_red = Pin(14, Pin.OUT)
buzzer = PWM(Pin(13)); buzzer.duty(0)

TANK_DEPTH_CM = 400
pressure_history = []


def read_distance_cm():
    trig.value(0); time.sleep_us(2)
    trig.value(1); time.sleep_us(10)
    trig.value(0)
    t0 = time.ticks_us()
    while echo.value() == 0:
        if time.ticks_diff(time.ticks_us(), t0) > 30000:
            return None
    start = time.ticks_us()
    while echo.value() == 1:
        if time.ticks_diff(time.ticks_us(), start) > 30000:
            return None
    end = time.ticks_us()
    return (time.ticks_diff(end, start) * 0.0343) / 2


def water_level_m():
    dist = read_distance_cm()
    if dist is None:
        if float_high.value(): return 3.6
        if float_mid.value():  return 2.2
        if float_low.value():  return 1.0
        return 0.05
    return max(0.0, min(4.0, (TANK_DEPTH_CM - dist) / 100.0))


def pressure_trend(hpa):
    pressure_history.append(hpa)
    if len(pressure_history) > 12:
        pressure_history.pop(0)
    if len(pressure_history) < 2:
        return 0.0
    return (pressure_history[-1] - pressure_history[0]) / len(pressure_history)


def buzz(freq, duty):
    if freq == 0:
        buzzer.duty(0)
    else:
        buzzer.freq(freq); buzzer.duty(duty)


def apply_outputs(state):
    led_green.value(1 if state == "SAFE" else 0)
    led_yellow.value(1 if state == "WARNING" else 0)
    if state == "CRITICAL":
        for _ in range(3):
            led_red.value(1); buzz(3500, 512); time.sleep_ms(120)
            led_red.value(0); buzz(0, 0);      time.sleep_ms(120)
    elif state == "WARNING":
        led_red.value(0)
        buzz(2000, 300); time.sleep_ms(100); buzz(0, 0)
    else:
        led_red.value(0); buzz(0, 0)


print("FloodSense node booting...")
lcd.show("FloodSense", "booting...")
time.sleep(1)

while True:
    water = water_level_m()
    rain = pot_rain.read_u16() * (200.0 / 65535.0)
    soil = pot_soil.read_u16() * (100.0 / 65535.0)
    hpa = bmp.read_hpa()
    trend = pressure_trend(hpa)

    try:
        dht_sensor.measure()
        humidity = dht_sensor.humidity()
        temp = dht_sensor.temperature()
    except Exception:
        humidity, temp = 0.0, 0.0

    state = "SAFE"
    reason = "all parameters nominal"

    if water >= 3.0:
        state, reason = "CRITICAL", "water >= 3.0m"
    elif water >= 2.0 and rain >= 180 and soil >= 98:
        state, reason = "CRITICAL", "saturated soil, heavy rain"
    elif water >= 2.0:
        state, reason = "WARNING", "water >= 2.0m"
    elif rain >= 90:
        state, reason = "WARNING", "rainfall >= 90mm"
    elif trend <= -1.0 and humidity > 85 and rain > 5:
        state, reason = "WARNING", "pressure falling, humid, raining"

    print('{"water_level_m": %.2f, "rainfall_mm": %.1f, "soil_pct": %.1f, '
          '"pressure_hpa": %.1f, "pressure_trend": %.2f, "humidity": %.1f, '
          '"temp_c": %.1f, "state": "%s", "reason": "%s"}' % (
              water, rain, soil, hpa, trend, humidity, temp, state, reason))

    if state == "SAFE":
        lcd.show("SYSTEM: SAFE", "Depth: %.2fm" % water)
    elif state == "WARNING":
        lcd.show("RISK: ELEVATED", reason[:16])
    else:
        lcd.show("!!! EVACUATE !!!", "Depth: %.2fm" % water)

    apply_outputs(state)
    time.sleep(1)
