(raw_clean_taps)

Used *.npz
  - actions
  - g: num_strikes (normalize by dividing max(num_strikes))

I used as follows:
  y = Query(Encoder(t, g, SM(t)))

