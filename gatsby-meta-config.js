module.exports = {
  title: 'seungbo-an.com', // seungbo-an.com
  description: 'AI Engineer',
  language: 'ko',
  siteUrl: 'https://www.seungbo-an.com', // https://www.seungbo-an.com
  ogImage: '/og-image.png', // 공유할 때 보이는 미리보기 이미지
  comments: {
    utterances: {
        repo: ''
    },
  },
  ga: '0', // Google Analytics Tracking ID
  author: {
    name: '안승보',
    bio: {
      description: ['AI Engineer', 'Frontend'],
      email: 'seungbo1112@gmail.com',
    },
    social: {
      github: 'https://github.com/seungboAn',
      linkedIn: 'https://www.linkedin.com/in/seungbo-an/',
    },
  },

  // metadata for About Page
  about: {
    timestamps: [
      // =====       [Timestamp Sample and Structure]      =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!) =====
      {
        date: '',
        activity: '',
        links: {
          github: '',
          post: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        date: '',
        activity: '',
        links: {
          post: '',
          github: 'https://github.com/seungboAn/',
          demo: 'https://www.seungbo-an.com',
        },
      },
    ],

    projects: [
      // =====        [Project Sample and Structure]        =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!)  =====
      {
        title: 'Portfolio',
        description: '블로그',
        techStack: ['gatsby', 'react'],
        thumbnailUrl: '', // Path to your in the 'assets' folder
          links: {
          post: '',
          github: '',
          demo: '',
          googlePlay: '',
          appStore: '',
          }
      }
      // ========================================================
      // ========================================================
    ],
  },
};
