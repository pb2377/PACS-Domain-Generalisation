# Code Sample

Small code sample for training a CNN model for PACS domain generalisation,
as well as some infrastructure to expand to DomainNet Generalisation. It will reproduced 
the baseline results that are reported in most papers looking at PACS DG.
 
Full readme TBD:
- Only included baseline models (i.e. training existing architectures as standard).
- Removed the extensive analysis code for now because there is nothing to compare against (this is only the baseline).
- Admin have changed something with the GPU cluster ports, so I removed tensorboard
 writing because I can't access it remotely atm.