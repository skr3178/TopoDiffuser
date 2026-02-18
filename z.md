 Training Set

   Sequence   Scene Type
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   00         City (long, complex)
   02         City (long, complex)
   05         City
   07         Residential
   Total      3,860 samples

  Test/Validation Set

   Sequence   Samples   Scene Type
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   08         1,391     Residential/commercial mixed
   09         530       Country/rural roads
   10         349       Urban city center
   Total      2,270     

  NOT Used

  • 01 - Highway (monotonous, less interesting)
  • 03 - Not available (test set overlap)
  • 04 - Short sequence
  • 06 - Loop/residential

  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Your Current Implementation

  Your train_diffusion_only.py has:

  train_sequences=['00', '02', '05', '07']  # ✅ Matches paper
  val_sequences=['08']                       # ⚠️ Only using 08, paper uses 08, 09, 10

  To match the paper exactly for validation, you should use:

  val_sequences=['08', '09', '10']