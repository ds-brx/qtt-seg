{
  "HP": [
    {
      "name": "batch_size",
      "type": "ordinal",
      "sequence": [
        32
       
      ],
      "default": 32
    },
    {
      "name": "lr",
      "type": "ordinal",
      "sequence": [
        1e-05,
        0.001,
        0.005,
        0.01
      ],
      "default": 1e-05
    },
    {
      "name": "layer_decay",
      "type": "ordinal",
      "sequence": [
        0.0,
        0.65,
        0.75
      ],
      "default": 0.0
    },
    {
      "name": "pct_to_freeze",
      "type": "ordinal",
      "sequence": [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8
      ],
      "default": 0.0
    },
    {
      "name": "model",
      "type": "catagorical",
      "sequence": [
        1
      ],
      "default": 0
    }
  ]
}
