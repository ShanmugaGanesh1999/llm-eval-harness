# Gold set v2

Changes from v1 (v1 is kept unchanged for comparison):

- `rag-005`: `key_facts` changed from `["missing approval ticket"]` to
  `["missing", "approval ticket"]`. The source sentence is "DEPLOY-4021 means the deployment is
  missing an approval ticket", so a correct extractive answer failed the v1 label. Found by the
  first live run against Project 1 (a labeling bug, not a system bug).
