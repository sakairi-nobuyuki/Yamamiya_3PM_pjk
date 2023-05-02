terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

provider "google" {
  credentials = file("yamamiya-pm-745bf58d9a45.json")

  project = "yamamiya-pm"
  region  = "us-central1"
  zone    = "us-central1-c"
}

# Create new storage bucket in the US multi-region
# with coldline storage
resource "random_id" "bucket_prefix" {
  byte_length = 8
}

resource "google_storage_bucket" "static" {
  name          = "yamamiya-pm-dataset"
  # name          = "${random_id.bucket_prefix.hex}-dataset"
  location      = "US"
  storage_class = "COLDLINE"

  uniform_bucket_level_access = true
}