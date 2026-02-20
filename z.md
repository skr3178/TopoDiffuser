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

Summary: OSM Alignment Workflow for Seq 01                                                                                  
                                                                                                                                
    Goal                                                                                                                        
                                                                                                                                
    Generate overlaid visualization of OSM road network aligned with KITTI trajectory for sequence 01.                          
                                                                                                                                
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ─────                                                                                                                         
    Input Files Used                                                                                                            
                                                                                                                                
     File          Path                                                       Purpose                                           
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━       
     OSM PBF       data/osm/karlsruhe.osm.pbf                                 Raw OpenStreetMap data (144MB, 260K roads)        
     KITTI Poses   data/kitti/poses/01.txt                                    Trajectory in local coordinates                   
     OXTS GPS      data/raw_data/2011_10_03_drive_0042_sync/oxts/data/*.txt   GPS lat/lon for alignment                         
                                                                                                                                
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ─────                                                                                                                         
    Programs Run (In Order)                                                                                                     
                                                                                                                                
    Step 1: Parse OSM PBF + Initial GPS Alignment                                                                               
                                                                                                                                
    python parse_full_osm_pbf.py --seq 01                                                                                       
                                                                                                                                
    What it does:                                                                                                               
                                                                                                                                
    • Loads Karlsruhe OSM PBF using pyrosm                                                                                      
    • Extracts driving network (2,750 road polylines)                                                                           
    • Loads OXTS GPS data and converts to UTM                                                                                   
    • Computes initial alignment:                                                                                               
      • Offset: UTM origin - trajectory origin                                                                                  
      • Rotation: GPS heading vs trajectory heading (~180°)                                                                     
    • Saves: osm_polylines_aligned_seq01.pkl (initial)                                                                          
                                                                                                                                
    Key insight: Initial alignment had 24m mean error - needed refinement.                                                      
                                                                                                                                
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ─────                                                                                                                         
    Step 2: Refine Rotation via Grid Search                                                                                     
                                                                                                                                
    python refine_osm_alignment.py --seq 01                                                                                     
                                                                                                                                
    What it does:                                                                                                               
                                                                                                                                
    • Loads initial polylines + trajectory                                                                                      
    • Searches rotation space (-45° to +45°) to minimize alignment error                                                        
    • Finds optimal adjustment: +5.5°                                                                                           
    • Applies rotation around trajectory center                                                                                 
    • Saves: osm_polylines_aligned_seq01_refined.pkl                                                                            
                                                                                                                                
    Result: Error reduced from 24.15m → 20.69m                                                                                  
                                                                                                                                
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ─────                                                                                                                         
    Step 3: Similarity Transform (Final Fix)                                                                                    
                                                                                                                                
    Manual Python script to fix start/end alignment:                                                                            
                                                                                                                                
    What it does:                                                                                                               
                                                                                                                                
    • Computes vector from OSM start→end and trajectory start→end                                                               
    • Calculates:                                                                                                               
      • Scale factor: 0.9887 (OSM data was 1.1% too large)                                                                      
      • Rotation: -0.25° (fine-tuning)                                                                                          
    • Applies: Translate → Scale → Rotate → Translate                                                                           
    • Saves final: osm_polylines_aligned_seq01.pkl                                                                              
                                                                                                                                
    Result: Start error = 0.00m, End error = 0.00m                                                                              
                                                                                                                                
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ─────                                                                                                                         
    Step 4: Generate Visualization                                                                                              
                                                                                                                                
    The scripts automatically generate visualization showing:                                                                   
                                                                                                                                
    • Top-left: Rotation search curve                                                                                           
    • Top-middle: OSM roads only                                                                                                
    • Top-right: Trajectory only                                                                                                
    • Bottom: Overlay with both start (green) and end (red star) aligned                                                        
                                                                                                                                
    Saves: osm_pbf_aligned_seq01.png and osm_pbf_aligned_seq01_refined.png                                                      
                                                                                                                                
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ─────                                                                                                                         
    Key Scripts Involved                                                                                                        
                                                                                                                                
     Script                          Purpose                                                                                    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                         
     parse_full_osm_pbf.py           Parse PBF + initial GPS alignment                                                          
     refine_osm_alignment.py         Grid search for optimal rotation                                                           
     utils/osm_alignment.py          latlon_to_utm(), load_oxts_data()                                                          
     utils/osm_polylines_to_bev.py   Polylines → BEV mask conversion          