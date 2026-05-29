//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! Mesh variables.
//! Prolongation of CC variables  occurs in ProlongateCC() function called from task list

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"   // MeshRefinement::RestrictCC (SyncParabolicGhosts)
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BValCC constructor:

MeshBoundaryValuesCC::MeshBoundaryValuesCC(MeshBlockPack *pp, ParameterInput *pin,
                                           bool z4c) :
  MeshBoundaryValues(pp, pin, z4c) {
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors.
//!
//! This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously
//! for ALL the MeshBlocks. This reduces the number of kernel launches when there are a
//! large number of MeshBlocks per MPI rank. Buffer data are then sent (via MPI) or copied
//! directly for periodic or block boundaries.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)
//! 5D Kokkos View of coarsened (restricted) array data also required with SMR/AMR

TaskStatus MeshBoundaryValuesCC::PackAndSendCC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      // if neighbor is at coarser level, use coar indices to pack buffer
      int il, iu, jl, ju, kl, ku;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = sbuf[n].icoar[0].bis;
        iu = sbuf[n].icoar[0].bie;
        jl = sbuf[n].icoar[0].bjs;
        ju = sbuf[n].icoar[0].bje;
        kl = sbuf[n].icoar[0].bks;
        ku = sbuf[n].icoar[0].bke;
      // if neighbor is at same level, use same indices to pack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = sbuf[n].isame[0].bis;
        iu = sbuf[n].isame[0].bie;
        jl = sbuf[n].isame[0].bjs;
        ju = sbuf[n].isame[0].bje;
        kl = sbuf[n].isame[0].bks;
        ku = sbuf[n].isame[0].bke;
      // if neighbor is at finer level, use fine indices to pack buffer
      } else {
        il = sbuf[n].ifine[0].bis;
        iu = sbuf[n].ifine[0].bie;
        jl = sbuf[n].ifine[0].bjs;
        ju = sbuf[n].ifine[0].bje;
        kl = sbuf[n].ifine[0].bks;
        ku = sbuf[n].ifine[0].bke;
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
      // in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy directly into recv buffer if MeshBlocks on same rank

        if (nghbr.d_view(m,n).rank == my_rank) {
          // if neighbor is at same or finer level, load data from u0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = ca(m,v,k,j,i);
            });
          }

        // else copy into send buffer for MPI communication below

        } else {
          // if neighbor is at same or finer level, load data from u0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = ca(m,v,k,j,i);
            });
          }
        }
      });
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer

  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // If neighbor is at same level and data is for Z4c module, append data from coarse
      // array for higher-order prolongation
      if ((nghbr.d_view(m,n).lev == mblev.d_view(m)) && (is_z4c) && (multilevel)) {
        il = sbuf[n].isame_z4c.bis;
        iu = sbuf[n].isame_z4c.bie;
        jl = sbuf[n].isame_z4c.bjs;
        ju = sbuf[n].isame_z4c.bje;
        kl = sbuf[n].isame_z4c.bks;
        ku = sbuf[n].isame_z4c.bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
        int ndat = nvar*sbuf[n].isame_ndat; // size of same level data already in buff

        // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
        // in MeshBlockPacks, so array index equals (target_id - first_id)
        int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        int dn = nghbr.d_view(m,n).dest;

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // Inner (vector) loop over i
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            // load data from coarse_u0
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              rbuf[dn].vars(dm,ndat+ (i-il + ni*(j-jl + nj*(k-kl + nk*v))))=ca(m,v,k,j,i);
            });

          // else copy into send buffer for MPI communication below
          } else {
            // load data from coarse_u0
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              sbuf[n].vars(m,ndat+ (i-il + ni*(j-jl + nj*(k-kl + nk*v))) )=ca(m,v,k,j,i);
            });
          }
        });
      }
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  auto &is_z4c = is_z4c_;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          int data_size = nvar;
          if ( nghbr.h_view(m,n).lev < pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= sendbuf[n].icoar_ndat;
          } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
            if (is_z4c) {
              data_size *= sendbuf[n].isame_z4c_ndat;
            } else {
              data_size *= sendbuf[n].isame_ndat;
            }
          } else {
            data_size *= sendbuf[n].ifine_ndat;
          }
          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus MeshBoundaryValuesCC::RecvAndUnpackCC(DvceArray5D<Real> &a,
                                                 DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &mblev = pmy_pack->pmb->mb_lev;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // if neighbor is at coarser level, use coar indices to unpack buffer
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = rbuf[n].icoar[0].bis;
        iu = rbuf[n].icoar[0].bie;
        jl = rbuf[n].icoar[0].bjs;
        ju = rbuf[n].icoar[0].bje;
        kl = rbuf[n].icoar[0].bks;
        ku = rbuf[n].icoar[0].bke;
      // if neighbor is at same level, use same indices to unpack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = rbuf[n].isame[0].bis;
        iu = rbuf[n].isame[0].bie;
        jl = rbuf[n].isame[0].bjs;
        ju = rbuf[n].isame[0].bje;
        kl = rbuf[n].isame[0].bks;
        ku = rbuf[n].isame[0].bke;
      // if neighbor is at finer level, use fine indices to unpack buffer
      } else {
        il = rbuf[n].ifine[0].bis;
        iu = rbuf[n].ifine[0].bie;
        jl = rbuf[n].ifine[0].bjs;
        ju = rbuf[n].ifine[0].bje;
        kl = rbuf[n].ifine[0].bks;
        ku = rbuf[n].ifine[0].bke;
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // if neighbor is at same or finer level, load data directly into u0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            a(m,v,k,j,i) = rbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) );
          });

        // if neighbor is at coarser level, load data into coarse_u0
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            ca(m,v,k,j,i) = rbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) );
          });
        }
      });
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // If neighbor is at same level and data is for Z4c module, unpack data from coarse
      // array for higher-order prolongation
      if ((nghbr.d_view(m,n).lev == mblev.d_view(m)) && (is_z4c) && (multilevel)) {
        il = rbuf[n].isame_z4c.bis;
        iu = rbuf[n].isame_z4c.bie;
        jl = rbuf[n].isame_z4c.bjs;
        ju = rbuf[n].isame_z4c.bje;
        kl = rbuf[n].isame_z4c.bks;
        ku = rbuf[n].isame_z4c.bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
        int ndat = nvar*rbuf[n].isame_ndat; // size of same level data packed in buff

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // load data into coarse_u0
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            ca(m,v,k,j,i) = rbuf[n].vars(m,ndat + (i-il + ni*(j-jl + nj*(k-kl + nk*v))) );
          });
        });
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesCC::SyncParabolicGhosts()
//! \brief Synchronous (blocking) per-substage neighbor ghost-zone refresh for an
//! operator-split parabolic operator (RKL2 super-time-stepping; ADR-0001, ADR-0009).
//!
//! The RKL2 substeps run INSIDE the integrator -- between the driver's normal boundary
//! task points -- so each substage needs a neighbor ghost-zone refresh of the diffused
//! field, across MeshBlocks and across MPI ranks (and, on a refined mesh, the
//! restriction/prolongation that fills coarse/fine boundary ghosts).  The driver tasklist
//! splits the cell-centered exchange (RestrictU -> InitRecv -> SendU -> RecvU ->
//! ClearSend/ClearRecv -> Prolongate) across tasks so async MPI overlaps; an
//! operator-split operator instead recomputes its flux every substage and so needs the
//! exchange to complete in ONE call.  This bundles the SAME existing routines
//! synchronously -- the variable-side analog of CorrectFlux -- adding NO new comms code:
//!   1. restrict the field into the coarse buffer (so coarser neighbors get restricted
//!      data) -- multilevel only;
//!   2. post receives, pack+send (direct copy into the neighbor's recv buffer on the same
//!      rank, MPI_Isend otherwise), busy-wait the receives, then clear sends/receives;
//!   3. fill the coarse boundary array from same-level neighbors and prolongate into the
//!      fine ghost zones -- multilevel only.
//! PHYSICAL domain boundaries (no neighbor: nghbr.gid < 0) are untouched here -- those
//! are the operator's own ApplyBoundary job (e.g. the insulated conduction fill).
//! Buffer width = a's 2nd extent (this object's InitializeBuffers(nvar)); the multigroup
//! FLD operator batches all groups into one exchange by sizing that width to all groups.
//! On a single MeshBlock with no refinement this is a no-op, so a single-block run is
//! byte-identical to the pre-#108 local-only path.

TaskStatus MeshBoundaryValuesCC::SyncParabolicGhosts(DvceArray5D<Real> &a,
                                                     DvceArray5D<Real> &ca) {
  int nvar = a.extent_int(1);
  auto &multilevel = pmy_pack->pmesh->multilevel;

  // (1) restrict interior into coarse buffer so coarser neighbors get restricted data
  if (multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(a, ca);
  }

  // (2) post receives, send, busy-wait receives to completion, clear (serial completes at
  // once; under MPI the non-blocking sends/recvs complete within this call)
  InitRecv(nvar);
  PackAndSendCC(a, ca);
  while (RecvAndUnpackCC(a, ca) == TaskStatus::incomplete) {}
  ClearRecv();
  ClearSend();

  // (3) fill coarse boundary array + prolongate into fine ghost zones at c/f faces
  if (multilevel) {
    FillCoarseInBndryCC(a, ca);
    ProlongateCC(a, ca);
  }
  return TaskStatus::complete;
}
