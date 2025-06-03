#$dataset_path = "C:\Users\Admin\Documents\face_recog\data\lfw_flat"
$dataset_path = "C:\Users\Admin\Documents\face_recog\data\lfw_raw"
$output_dir = "C:\Users\Admin\Documents\face_recog\embeddings"
$openface_path = "C:\Users\Admin\Documents\OpenFace_2.2.0_win_x64"

#cd $openface_path
#.\FeatureExtraction.exe -fdir $dataset_path -out_dir $output_dir 
#.\FeatureExtraction.exe -fdir $dataset_path -out_dir $output_dir -2Dfp -pose

# Aller dans le répertoire de OpenFace
Set-Location $openface_path

# Boucler sur chaque sous-dossier (personne)
Get-ChildItem -Path $dataset_path -Directory | ForEach-Object {
    $person_dir = $_.FullName
    $person_name = $_.Name
    $person_output = Join-Path $output_dir $person_name

    # Créer le dossier de sortie pour cette personne
    if (!(Test-Path $person_output)) {
        New-Item -ItemType Directory -Path $person_output | Out-Null
    }

    # Exécuter l’extraction
    .\FeatureExtraction.exe -fdir $person_dir -out_dir $person_output -2Dfp -pose
}
