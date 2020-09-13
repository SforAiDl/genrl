python -m pip install --upgrade pip

Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip
{
    param([string]$zipfile, [string]$outpath)

    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

# Invoke-WebRequest -uri http://prdownloads.sourceforge.net/swig/swigwin-4.0.2.zip -Method "GET" -OutFile "D:\a\swigwin-4.0.2.zip"
# Unzip "D:\a\swigwin-4.0.2.zip" "D:\a\swigwin-4.0.2"
# copy-item -path "D:\a\swigwin-4.0.2\swig.exe" -destination ".\"

(New-Object System.Net.WebClient).DownloadFile("http://prdownloads.sourceforge.net/swig/swigwin-4.0.1.zip","swigwin-4.0.1.zip");
Expand-Archive .\swigwin-4.0.1.zip .;
Copy-item -path ".\swigwin-4.0.1\swig.exe" -destination ".\"
$env:Path += ";D:\a\genrl\genrl\"

pip install torch==1.4.0 --quiet --find-links https://download.pytorch.org/whl/cpu/torch_stable.html --upgrade
pip install -r requirements.txt
