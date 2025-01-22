<h1 align="center">
  Seungbo An Gatsby Blog Template
</h1>

<p align="center">
  <a href="https://github.com/seungboAn/seungbo-an-gatsby-blog-template/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-0BSD-blue.svg" alt="Seungbo An gatsby blog template is released under the 0BSD license." />
  </a>
</p>

#### Demo Websites: [Seungbo An](https://www.seungb-an.com)
#### Source: [Github](https://github.com/seungboAn/seungbo-an-gatsby-blog-template)

## 👋 소개

매 주 목요일에 모여서 글쓰기 스터디를 진행하기 위해 다양한 플랫폼을 고민해봤는데요. 프론트엔드를 활용해보기 위해 직접 블로그를 개발하기 위해 Gatsby로 블로그를 시작하게되었습니다.

## ✨ 기능

- 😛 미모지와 문자 애니메이션를 통한 자기 소개
- 🔍 포스팅 검색 지원
- 🌘 다크모드 지원
- 💅 코드 하이라이팅 지원
- 👉 글 목차 자동 생성(ToC)
- 💬 Utterances 댓글 기능 지원
- ⚙️ meta-config를 통한 세부 설정 가능
- 👨‍💻 About Page 내용 변경 가능
- 📚 Posts Page 자동 생성
- 🛠 sitemap.xml, robots.txt 자동 생성
- 📈 Google Analytics 지원
- 🧢 Emoji 지원


## 🚀 시작하기

Github Page, Netlify 중 원하시는 배포 환경으로 블로그를 만드실 수 있습니다.

### 🔧 Netlify로 만들기

아래 버튼을 활용하면 개인 계정에 ``를 사용하고 있는 Repository 생성과 Netlify에 배포를 동시에 진행할 수 있습니다. 이후에, 생성된 Repository를 clone합니다.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/seungboAn/seungbo-an-gatsby-blog-template)

### 🏃‍♀️ 실행하기

아래 명령어를 실행하여 로컬 환경에 블로그를 실행합니다.

```bash
# Install dependencies
$ npm install

# Start development server
# http://localhost:8000
$ npm start
```

<br/>

## ⚙️ 블로그 정보 입력하기

`gatsby-meta-config.js`에 있는 정보를 수정해줍니다.

### 1. 블로그 기본 정보

```js
title: 'seungbo-an.com', // seungbo-an.com
description: 'AI Engineer',
language: 'ko',
siteUrl: 'https://www.seungbo-an.com', // https://www.seungbo-an.com
ogImage: '/og-image.png', // 공유할 때 보이는 미리보기 이미지
```

### 2. 댓글 설정

블로그 글들에 댓글을 달 수 있길 원하신다면 utterances를 통해서 이를 설정하실 수 있습니다.

> 🦄 utterances 사용방법은 [링크](https://utteranc.es/)를 참고해주세요!

```js
comments: {
    utterances: {
        repo: ''
    },
}
```

### 3. 글쓴이 정보

글쓴이(author)에 입력하신 정보는 홈페이지와 about 페이지 상단에 사용됩니다.

```js
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
```

### 4. timestamps

아래와 같이 각 timestamp 정보를 배열로 제공해주시면 입력하신 순서에 맞춰서 timestamps section에 보여지게 됩니다.

> links에 해당 정보가 없다면 생략해도 됩니다.

```js
{
  date: '',
  activity: '',
  links: {
    post: '/gatsby-starter-zoomkoding-introduction',
    github: 'https://github.com/seungboAn/',
    demo: 'https://www.seungbo-an.com',
  },
},
```

### 5. projects

마찬가지로 각 project 정보를 배열로 제공해주시면 입력하신 순서에 맞춰서 projects section에 보여지게 됩니다.

```js
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
```

<br/>

그렇게 내용을 문제 없이 입력하셨다면 나만의 블로그가 탄생한 것을 확인하실 수 있습니다.🎉

> `gatsby-meta-config.js`를 수정하셨다면, `npm start`를 통해 재실행해주세요!

## ✍️ 글 쓰기

본격적으로 블로그에 글을 쓰려면 `/content` 아래에 디렉토리를 생성하고 `index.md`에 markdown으로 작성하시면 됩니다.

> 폴더의 이름은 경로를 생성하는데 됩니다.

### 🏗 메타 정보

index.md 파일의 상단에는 아래와 같이 emoji, title, date, author, tags, categories 정보를 제공해야 합니다.

> emoji는 글머리에 보여지게 되며, categories는 띄어쓰기로 구분해서 입력할 수 있습니다.

```
---
emoji: ⚒️
title: 기술 블로그 직접 개발하기
date: '2024-12-27'
author: seungbo An
tags: Dev
categories: Dev
---
```

### 🖼 이미지 경로

글에 이미지를 첨부하고 싶으시다면 같은 디렉토리에 이미지 파일을 추가하셔서 아래와 같이 사용하시면 됩니다.

```
![사진](./[이미지 파일명])
```

### 🔍 목차 생성

글의 우측에 목차가 보이기를 원하신다면 `index.md` 파일 맨 아래에 다음 내용을 추가하시면 자동으로 목차가 생성됩니다.

    ```toc
    ```

### 문의

궁금하신 점이 있으시다면 [여기]()에 남겨주세요.