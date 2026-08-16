#!/usr/bin/env bash
#
# Downloads and unpacks one DE-SynPUF subsample from CMS.
#
# Sample 2 is the default. The CMS page for sample 1 links to the sample 20 file for
# the 2010 beneficiary summary, and no 2010 summary file for sample 1 is published at
# the canonical path, so sample 1 cannot support a 2009 to 2010 prospective split.
# Every subsample is an equivalent random draw of roughly 116,000 beneficiaries.

set -euo pipefail

SAMPLE="${1:-2}"
RAW_DIR="data/raw"
UNZIP_DIR="data/unzipped"

CMS_BASE="https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads"
CMS_FILES="https://downloads.cms.gov/files"

mkdir -p "${RAW_DIR}" "${UNZIP_DIR}"

URLS=(
  "${CMS_BASE}/de1_0_2008_beneficiary_summary_file_sample_${SAMPLE}.zip"
  "${CMS_BASE}/de1_0_2009_beneficiary_summary_file_sample_${SAMPLE}.zip"
  "${CMS_BASE}/de1_0_2010_beneficiary_summary_file_sample_${SAMPLE}.zip"
  "${CMS_BASE}/de1_0_2008_to_2010_inpatient_claims_sample_${SAMPLE}.zip"
  "${CMS_BASE}/de1_0_2008_to_2010_outpatient_claims_sample_${SAMPLE}.zip"
  "${CMS_FILES}/DE1_0_2008_to_2010_Carrier_Claims_Sample_${SAMPLE}A.zip"
  "${CMS_FILES}/DE1_0_2008_to_2010_Carrier_Claims_Sample_${SAMPLE}B.zip"
  "${CMS_FILES}/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_${SAMPLE}.zip"
)

for url in "${URLS[@]}"; do
  name="$(basename "${url}")"
  if [ -f "${RAW_DIR}/${name}" ]; then
    echo "already downloaded ${name}"
  else
    echo "downloading ${name}"
    curl -sSL --fail --max-time 2400 -A "Mozilla/5.0" -o "${RAW_DIR}/${name}" "${url}"
  fi
  unzip -o -q "${RAW_DIR}/${name}" -d "${UNZIP_DIR}/"
done

echo "extract ready in ${UNZIP_DIR}"
ls -la "${UNZIP_DIR}"
