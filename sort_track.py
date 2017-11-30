"""
    track car and etc based on sort
    Copyright (C) Willi

"""
from sort.sort import *

class TrackSort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, messes):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
          pos = self.trackers[t].predict()[0]
          trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
          if(np.any(np.isnan(pos))):
            to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
          self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
          if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0],0]
            trk.update(dets[d,:][0])

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            #trk = KalmanBoxTracker(dets[i,:])
            trk = ObjectTracker(dets[i,:], messes[i])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(trk)
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
              self.trackers.pop(i)
        return ret

    def set_tracker(self, new_tracks):
        self.trackers = new_tracks

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # all train
    sequences = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2',
                 'KITTI-13', 'KITTI-17', 'ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2']
    args = parse_args()
    display = args.display
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if (display):
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()

    if not os.path.exists('output'):
        os.makedirs('output')

    for seq in sequences:
        mot_tracker = TrackSort()  # create instance of the SORT tracker
        seq_dets = np.loadtxt('data/%s/det.txt' % (seq), delimiter=',')  # load detections
        with open('output/%s.txt' % (seq), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                j = 0
                messes = []
                while j < np.size(dets,0):
                    messes.append('test')
                    j += 1

                if (display):
                    ax1 = fig.add_subplot(111, aspect='equal')
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets, messes)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for trk in trackers:
                    d = trk.get_state()[0]
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, trk.id, d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[trk.id % 32, :]))
                        ax1.set_adjustable('box-forced')

                if (display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    if (display):
        print("Note: to get real runtime results run without the option: --display")
